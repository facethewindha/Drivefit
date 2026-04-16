# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""

from pytorch_fid import fid_score


import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
import torch.nn.functional as F
import re

os.environ["NCCL_SOCKET_IFNAME"] = "enp0s31f6"
os.environ["NCCL_IB_DISABLE"] = "1"

from drivefit_models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from inversion import create_inverter

import wandb
import lpips
from dataset import CustomImageFolder, PairedWeatherDataset, PairedWeatherReconDataset
from editing.conditioning import WeatherEditWrapper
from editing.seg_extractor import FrozenSegFormer
from editing.edge_ops import sobel_edges
os.environ["WANDB_MODE"] = "offline"


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def extract_task_specific_parameters(model):
    task_specific_parameters = {}
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            task_specific_parameters[name] = param
    return task_specific_parameters


def requires_grad(model, flag=True, depth=28):
    # WeatherEditWrapper compatibility: actual DiT backbone lives in `model.backbone`.
    core_model = model.backbone if hasattr(model, "backbone") else model
    core_modulation = getattr(core_model, "modulation", False)

    if core_modulation:
        for name, param in model.named_parameters():
            if not flag:
                param.requires_grad = False
                continue

            # 1) New wrapper modules (not under backbone) stay trainable.
            if not name.startswith("backbone."):
                param.requires_grad = True
                continue

            # 2) Backbone keeps original modulation fine-tuning policy.
            if ("weight_modulation" in name) or ("scenario_embedding_table" in name):
                # Restrict trainable depth inside blocks.
                m = re.search(r"blocks\.(\d+)\.", name)
                if m is not None:
                    block_idx = int(m.group(1))
                    param.requires_grad = block_idx < depth
                else:
                    # Non-block params (e.g. embeddings) stay trainable.
                    param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for _, param in model.named_parameters():
            param.requires_grad = flag


def init_weather_embed_from_ssei(
    model,
    embed_ckpt_path,
    source_weather="sunny",
    target_weather="rain",
    logger=None,
):
    """
    Initialize src/tgt weather scenario embeddings from SSEI checkpoint.
    Expected key: y_embedder.scenario_embedding_table.weight, shape [K, D]
    We map rows by WEATHER_TO_IDX and use the last row as null.
    """
    if embed_ckpt_path is None:
        return

    ckpt = torch.load(embed_ckpt_path, map_location="cpu")
    key = "y_embedder.scenario_embedding_table.weight"
    if key not in ckpt:
        if logger is not None:
            logger.warning(f"[SSEI] key not found: {key} in {embed_ckpt_path}")
        return

    w = ckpt[key]
    if w.ndim != 2 or w.shape[0] < 3:
        if logger is not None:
            logger.warning(f"[SSEI] unexpected shape: {tuple(w.shape)}")
        return

    weather_to_idx = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}
    if source_weather not in weather_to_idx or target_weather not in weather_to_idx:
        if logger is not None:
            logger.warning(
                f"[SSEI] unsupported weather names: source={source_weather}, target={target_weather}"
            )
        return

    src_idx = weather_to_idx[source_weather]
    tgt_idx = weather_to_idx[target_weather]
    null_idx = w.shape[0] - 1
    max_valid = w.shape[0] - 2
    if src_idx > max_valid or tgt_idx > max_valid:
        if logger is not None:
            logger.warning(
                f"[SSEI] weather index out of range for checkpoint rows: "
                f"src_idx={src_idx}, tgt_idx={tgt_idx}, max_valid={max_valid}"
            )
        return

    selected = torch.stack([w[src_idx], w[tgt_idx], w[null_idx]], dim=0).clone()

    m = model.backbone if hasattr(model, "backbone") else model
    ok = (
        hasattr(m, "src_weather_embedder")
        and hasattr(m, "tgt_weather_embedder")
        and hasattr(m.src_weather_embedder, "scenario_embedding_table")
        and hasattr(m.tgt_weather_embedder, "scenario_embedding_table")
    )
    if not ok:
        if logger is not None:
            logger.warning("[SSEI] src/tgt weather embedders not found on model.")
        return

    src_w = m.src_weather_embedder.scenario_embedding_table.weight
    tgt_w = m.tgt_weather_embedder.scenario_embedding_table.weight
    if src_w.shape != selected.shape or tgt_w.shape != selected.shape:
        if logger is not None:
            logger.warning(
                f"[SSEI] shape mismatch: src={tuple(src_w.shape)}, tgt={tuple(tgt_w.shape)}, "
                f"selected={tuple(selected.shape)}"
            )
        return

    with torch.no_grad():
        src_w.copy_(selected)
        tgt_w.copy_(selected)

    if logger is not None:
        logger.info(
            f"[SSEI] initialized weather embeddings from {embed_ckpt_path} "
            f"using rows src={src_idx}({source_weather}), tgt={tgt_idx}({target_weather}), null={null_idx}"
        )


def get_stage2_mix_weights(train_steps, args):
    """
    Linear annealing for mixed src/tgt supervision in stage2.
    Start from (w_src_start, w_tgt_start), anneal to (w_src_end, w_tgt_end).
    """
    start = int(args.mix_anneal_start_step)
    end = int(args.mix_anneal_end_step)
    if end <= start:
        return float(args.w_src_end), float(args.w_tgt_end)
    if train_steps <= start:
        return float(args.w_src_start), float(args.w_tgt_start)
    if train_steps >= end:
        return float(args.w_src_end), float(args.w_tgt_end)

    r = (train_steps - start) / float(end - start)
    w_src = float(args.w_src_start) + r * (float(args.w_src_end) - float(args.w_src_start))
    w_tgt = float(args.w_tgt_start) + r * (float(args.w_tgt_end) - float(args.w_tgt_start))
    return w_src, w_tgt


def sample_stage2_timesteps(batch_size, device, args):
    """
    Stage2 timestep sampler:
    - mostly from mid range [edit_timestep_min, edit_timestep_max]
    - a small portion from low range [stage2_low_t_min, stage2_low_t_max]
    """
    mid_min = int(args.edit_timestep_min)
    mid_max = int(args.edit_timestep_max)
    low_min = int(args.stage2_low_t_min)
    low_max = int(args.stage2_low_t_max)
    low_prob = float(args.stage2_low_t_prob)

    # Keep ranges valid and inclusive.
    if mid_max < mid_min:
        mid_min, mid_max = mid_max, mid_min
    if low_max < low_min:
        low_min, low_max = low_max, low_min

    t_mid = torch.randint(mid_min, mid_max + 1, (batch_size,), device=device)
    t_low = torch.randint(low_min, low_max + 1, (batch_size,), device=device)
    if low_prob <= 0.0:
        return t_mid
    if low_prob >= 1.0:
        return t_low

    choose_low = torch.rand(batch_size, device=device) < low_prob
    return torch.where(choose_low, t_low, t_mid)


def load_partial_checkpoint(model, state_dict, logger, tag="", try_backbone_prefix=False):
    """
    Load checkpoint with diagnostics.
    - strips optional 'module.' prefix
    - optional retry by adding 'backbone.' prefix (for wrapped models)
    """
    if state_dict is None:
        return 0

    state = dict(state_dict)
    if any(k.startswith("module.") for k in state.keys()):
        state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}

    model_state = model.state_dict()

    def compatible_keys(sd):
        keys = []
        for k, v in sd.items():
            if k in model_state and hasattr(v, "shape") and model_state[k].shape == v.shape:
                keys.append(k)
        return keys

    comp = compatible_keys(state)

    if try_backbone_prefix and len(comp) == 0 and hasattr(model, "backbone"):
        remap = {}
        for k, v in state.items():
            k2 = k
            if not k2.startswith("backbone."):
                cand = f"backbone.{k2}"
                if cand in model_state and hasattr(v, "shape") and model_state[cand].shape == v.shape:
                    remap[cand] = v
                    continue
            if k2 in model_state and hasattr(v, "shape") and model_state[k2].shape == v.shape:
                remap[k2] = v
        if len(remap) > 0:
            state = remap
            comp = compatible_keys(state)

    ret = model.load_state_dict(state, strict=False)
    if logger is not None:
        logger.info(
            f"[{tag}] compatible={len(comp)}, missing={len(ret.missing_keys)}, unexpected={len(ret.unexpected_keys)}"
        )
        if len(ret.unexpected_keys) > 0:
            logger.info(f"[{tag}] unexpected (head): {ret.unexpected_keys[:8]}")
        if len(ret.missing_keys) > 0:
            logger.info(f"[{tag}] missing (head): {ret.missing_keys[:8]}")
    return len(comp)


def calculate_params_num(model, rank):
    """
    Calculate the number of parameters in a model.
    """
    tune_param_num = 0
    origin_param_num = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            tune_param_num += torch.numel(param)
        else:
            origin_param_num += torch.numel(param)
    return tune_param_num, origin_param_num


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################
#
def compute_edge_loss(img1, img2):
    """Sobel edge-consistency loss for two images in shape (B,3,H,W), range [-1,1]."""
    import torch.nn.functional as F
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=img1.dtype, device=img1.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    gray1 = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
    gray2 = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
    gray1 = gray1.unsqueeze(1)
    gray2 = gray2.unsqueeze(1)
    e1 = (F.conv2d(gray1, kx, padding=1) ** 2 + F.conv2d(gray1, ky, padding=1) ** 2).sqrt()
    e2 = (F.conv2d(gray2, kx, padding=1) ** 2 + F.conv2d(gray2, ky, padding=1) ** 2).sqrt()
    return F.l1_loss(e1, e2)


def main(args):

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group("gloo")

    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}, device={device}."
    )

    if rank == 0:
        log = wandb.init(project="DriveDiTFit", resume=False, config=args)

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        results_dir = f"{args.results_dir}/{args.dataset_name}"
        os.makedirs(results_dir, exist_ok=True)
        experiment_index = len(glob(f"{results_dir}/*"))+1
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    logger.info(args)
    logger.info("visible device: " + os.environ["CUDA_VISIBLE_DEVICES"])
    if args.task_type == "image_editing_paired":
        logger.info(
            "Timestep policy: stage1 full-range [0, T-1], "
            f"stage2 mid [{args.edit_timestep_min}, {args.edit_timestep_max}] + "
            f"low [{args.stage2_low_t_min}, {args.stage2_low_t_max}] prob={args.stage2_low_t_prob}"
        )
    
    # oggerheckpoint
    resume_epoch = 0
    resume_train_steps = 0
    
    logger.info(f"[DEBUG] args.resume_checkpoint = {args.resume_checkpoint}")

    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    # model = DiT_models[args.model](
    #     input_size=latent_size,
    #     num_classes=args.num_classes,
    #     modulation=args.modulation,
    #     patch_modulation=args.patch_modulation,
    #     block_mlp_modulation=args.block_mlp_modulation,
    #     cond_mlp_modulation=args.cond_mlp_modulation,
    #     rank=args.rank,
    #     #
    #     use_src_cond=(args.task_type == "image_editing"),
    #     # 
    #     # scenario_num=args.scenario_num if args.dataset_name is not None else 0,
    #     scenario_num=2 if args.task_type == "image_editing" else 
    #     (args.scenario_num if args.dataset_name is not None else 0),

    #     rope=args.rope,
    #     finetune_depth=args.finetune_depth,
    # )
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        modulation=args.modulation,
        patch_modulation=args.patch_modulation,
        block_mlp_modulation=args.block_mlp_modulation,
        cond_mlp_modulation=args.cond_mlp_modulation,
        rank=args.rank,
        use_src_cond=False,  # Paired editing uses 4-channel latent input only.
        scenario_num=2 if args.task_type == "image_editing_paired" else (args.scenario_num if args.dataset_name is not None else 0),
        rope=args.rope,
        finetune_depth=args.finetune_depth,
    )

    # 1) Load pretrained base model weights.
    if args.resume_checkpoint is not None:
        logger.info(f"Loading pretrained model from: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        state_dict = checkpoint["model"] if (isinstance(checkpoint, dict) and "model" in checkpoint) else checkpoint
        load_partial_checkpoint(model, state_dict, logger, tag="resume_checkpoint")
        logger.info("Loaded pretrained base model")

    # Wrap model for paired editing condition injection.
    if args.task_type == "image_editing_paired":
        model = WeatherEditWrapper(
            backbone=model,
            seg_channels=19,
            d_model=model.pos_embed.shape[-1],
            num_heads=model.num_heads,
        )
    else:
        model = model

    # 2) Optionally load fine-tuned modulation checkpoint (after wrapper construction).
    if args.modulation_checkpoint is not None:
        args.modulation_checkpoint = args.modulation_checkpoint.strip()
        logger.info(f"Loading fine-tuned checkpoint from: {args.modulation_checkpoint}")
        checkpoint = torch.load(args.modulation_checkpoint, map_location="cpu")
        state_dict = checkpoint["model"] if (isinstance(checkpoint, dict) and "model" in checkpoint) else checkpoint
        _ = load_partial_checkpoint(
            model,
            state_dict,
            logger,
            tag="modulation_checkpoint",
            try_backbone_prefix=(args.task_type == "image_editing_paired"),
        )
        resume_epoch = checkpoint.get("epoch", 0) if isinstance(checkpoint, dict) else 0
        resume_train_steps = checkpoint.get("train_steps", 0) if isinstance(checkpoint, dict) else 0
        logger.info(f"Loaded fine-tuned params, resuming from epoch {resume_epoch}, step {resume_train_steps}")
    # 3) Load / initialize weather embeddings.
    if args.embed_checkpoint is not None:
        if args.task_type == "image_editing_paired":
            init_weather_embed_from_ssei(
                model,
                args.embed_checkpoint,
                source_weather=args.source_weather,
                target_weather=args.target_weather,
                logger=logger,
            )
        else:
            checkpoint = torch.load(args.embed_checkpoint, map_location="cpu")
            model.load_state_dict(checkpoint, strict=False)
    requires_grad(model, True, args.finetune_depth)
    if args.task_type == "image_editing_paired":
        core_model = model.backbone if hasattr(model, "backbone") else model
        if hasattr(core_model, "src_weather_embedder"):
            for p in core_model.src_weather_embedder.parameters():
                p.requires_grad = False
            logger.info("Froze src_weather_embedder parameters.")
    seg_extractor = None
    lpips_fn = None
    if args.task_type == "image_editing_paired":
        seg_extractor = FrozenSegFormer(
            model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            out_size=(args.image_size, args.image_size),
        ).to(device)
        lpips_fn = lpips.LPIPS(net="alex").to(device)
        lpips_fn.eval()
        for p in lpips_fn.parameters():
            p.requires_grad = False

    model = DDP(model.to(device), device_ids=[device])#,find_unused_parameters=True)
    

    vae = AutoencoderKL.from_pretrained(args.vae_checkpoint).to(device)
    modulation_params_num, origin_params_num = calculate_params_num(model, rank)
    if rank == 0:
        log.config.update(
            {
                "modulation_params_num": modulation_params_num,
                "origin_params_num": origin_params_num,
            }
        )
    logger.info(
        f"DriveDiTFit Trainable Parameters: {modulation_params_num}\n DiT Frozen Parameters: {origin_params_num}"
    )

    if args.task_type == "image_editing_paired":
        dit_params = []
        new_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("backbone."):
                dit_params.append(p)
            else:
                new_params.append(p)

        opt = torch.optim.AdamW(
            [
                {"params": dit_params, "lr": args.lr_dit},
                {"params": new_params, "lr": args.lr_new},
            ],
            weight_decay=1e-2,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # 
    transform = transforms.Compose( 
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    # dataset
    if args.task_type == "image_editing_paired":
        if args.train_stage == "stage1":
            # Stage1 uses 1225 pairs expanded to 2450 single-weather recon samples.
            dataset = PairedWeatherReconDataset(
                data_root=args.weather_data_path,
                source_weather=args.source_weather,
                target_weather=args.target_weather,
                transform=transform,
                max_pairs=1225,
            )
        else:
            # Stage2 uses paired train split for sunny->rain editing.
            dataset = PairedWeatherDataset(
                data_root=args.weather_data_path,
                source_weather=args.source_weather,
                target_weather=args.target_weather,
                transform=transform,
                max_pairs=1225,
                split="train",
                train_count=980,
                val_count=122,
                test_count=123,
            )
    else:
        if args.boxes_path is None:
            dataset = ImageFolder(args.data_path, transform=transform)
        else:
            dataset = CustomImageFolder(
            args.data_path, args.boxes_path, trans_flip=True, transform=transform
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"Dataset contains {len(dataset):,} images ({args.weather_data_path})")
    logger.info(f"Gradient accumulation steps: {args.grad_accum_steps}")

    model.train()

    train_steps = resume_train_steps
    log_steps = 0
    running_loss = 0
    start_time = time()

    interval = args.epochs // 6

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(resume_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        accum_steps = max(1, int(args.grad_accum_steps))
        accum_counter = 0
        opt.zero_grad(set_to_none=True)

        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=args.noise_schedule,
            flag=epoch // interval,
        )
        # datadataset
        for data in loader:
            if args.task_type == "image_editing_paired":
                m = model.module if hasattr(model, "module") else model
                core = m.backbone if hasattr(m, "backbone") else m

                if args.train_stage == "stage1":
                    # Stage1: one-weather reconstruction per step.
                    img = data["img"].to(device)
                    y_recon = (
                        torch.tensor(data["label"], device=device).long()
                        if not isinstance(data["label"], torch.Tensor)
                        else data["label"].to(device).long()
                    )
                    with torch.no_grad():
                        z_recon = vae.encode(img).latent_dist.sample().mul_(0.18215)

                    bsz = z_recon.shape[0]
                    # Stage1 uses full diffusion range.
                    t_recon = torch.randint(0, diffusion.num_timesteps, (bsz,), device=device)
                    noise_recon = torch.randn_like(z_recon)
                    z_noisy = diffusion.q_sample(z_recon, t_recon, noise=noise_recon)
                    # Stage1 intentionally removes structure conditions and uses only weather label.
                    model_out = core(
                        z_noisy,
                        t_recon,
                        y_tgt=y_recon,
                    )
                    c_latent = z_recon.shape[1]
                    pred_noise, model_var_values = torch.split(model_out, c_latent, dim=1)
                    L_diff = F.mse_loss(pred_noise, noise_recon)
                    L_var = torch.tensor(0.0, device=device)
                    if core.learn_sigma:
                        frozen_out = torch.cat([pred_noise.detach(), model_var_values], dim=1)
                        vb_terms = diffusion._vb_terms_bpd(
                            model=lambda *args, r=frozen_out: r,
                            x_start=z_recon,
                            x_t=z_noisy,
                            t=t_recon,
                            clip_denoised=False,
                        )
                        L_var = vb_terms["output"]
                        if diffusion.loss_type.name == "RESCALED_MSE":
                            L_var *= diffusion.num_timesteps / 1000.0
                    z0_hat = diffusion._predict_xstart_from_eps(z_noisy, t_recon, pred_noise)
                    L_latent = F.l1_loss(z0_hat, z_recon)
                    L_app = torch.tensor(0.0, device=device)
                    L_edge = torch.tensor(0.0, device=device)
                    L_seg = torch.tensor(0.0, device=device)
                    w_src = 1.0
                    w_tgt = 0.0
                    loss = L_diff + args.lambda_latent * L_latent + args.lambda_var * L_var
                else:
                    # Stage2: sunny->rain editing with mixed src/tgt diffusion supervision.
                    src_img = data["src_img"].to(device)
                    tgt_img = data["tgt_img"].to(device)
                    y_src = (
                        torch.tensor(data["src_label"], device=device).long()
                        if not isinstance(data["src_label"], torch.Tensor)
                        else data["src_label"].to(device).long()
                    )
                    y_tgt = (
                        torch.tensor(data["tgt_label"], device=device).long()
                        if not isinstance(data["tgt_label"], torch.Tensor)
                        else data["tgt_label"].to(device).long()
                    )
                    with torch.no_grad():
                        z_src = vae.encode(src_img).latent_dist.sample().mul_(0.18215)
                        z_tgt = vae.encode(tgt_img).latent_dist.sample().mul_(0.18215)
                        seg_src = seg_extractor(src_img)
                    edge_src = sobel_edges(src_img)

                    bsz = z_src.shape[0]
                    # Stage2 uses mixed timestep sampling (mid-heavy + low-step minority).
                    t = sample_stage2_timesteps(bsz, device, args)
                    noise_src = torch.randn_like(z_src)
                    z_s_noisy = diffusion.q_sample(z_src, t, noise=noise_src)
                    model_out = model(
                        z_s_noisy,
                        t,
                        y_tgt=y_tgt,
                        src_img=src_img,
                        seg_logits=seg_src,
                        edge_map=edge_src,
                    )

                    c_latent = z_src.shape[1]
                    pred_noise, model_var_values = torch.split(model_out, c_latent, dim=1)

                    eps_src = noise_src
                    eps_tgt = diffusion._predict_eps_from_xstart(z_s_noisy, t, z_tgt)
                    w_src, w_tgt = get_stage2_mix_weights(train_steps, args)
                    L_diff_src = F.mse_loss(pred_noise, eps_src)
                    L_diff_tgt = F.mse_loss(pred_noise, eps_tgt)
                    L_diff = w_src * L_diff_src + w_tgt * L_diff_tgt

                    L_var = torch.tensor(0.0, device=device)
                    if core.learn_sigma:
                        frozen_out = torch.cat([pred_noise.detach(), model_var_values], dim=1)
                        vb_src = diffusion._vb_terms_bpd(
                            model=lambda *args, r=frozen_out: r,
                            x_start=z_src,
                            x_t=z_s_noisy,
                            t=t,
                            clip_denoised=False,
                        )["output"]
                        vb_tgt = diffusion._vb_terms_bpd(
                            model=lambda *args, r=frozen_out: r,
                            x_start=z_tgt,
                            x_t=z_s_noisy,
                            t=t,
                            clip_denoised=False,
                        )["output"]
                        L_var = w_src * vb_src + w_tgt * vb_tgt
                        if diffusion.loss_type.name == "RESCALED_MSE":
                            L_var *= diffusion.num_timesteps / 1000.0

                    z0_hat = diffusion._predict_xstart_from_eps(z_s_noisy, t, pred_noise)
                    L_latent = F.l1_loss(z0_hat, z_tgt)
                    x_hat = vae.decode(z0_hat / 0.18215).sample
                    x_hat = torch.clamp(x_hat, -1.0, 1.0)
                    L_app_l1 = F.l1_loss(x_hat, tgt_img)
                    with torch.no_grad():
                        seg_hat = seg_extractor(x_hat)
                    L_app_lpips = lpips_fn(x_hat, tgt_img).mean()
                    L_app = L_app_l1 + args.lambda_lpips * L_app_lpips
                    edge_hat = sobel_edges(x_hat)
                    L_edge = F.l1_loss(edge_hat, edge_src)
                    L_seg = F.l1_loss(seg_hat, seg_src)
                    loss = (
                        L_diff
                        + args.lambda_latent * L_latent
                        + args.lambda_app * L_app
                        + args.lambda_edge * L_edge
                        + args.lambda_seg * L_seg
                        + args.lambda_var * L_var
                    )
                loss_dict = {
                    "loss": loss,
                    "L_diff": L_diff.detach(),
                    "L_latent": L_latent.detach(),
                    "L_app": L_app.detach(),
                    "L_edge": L_edge.detach(),
                    "L_seg": L_seg.detach(),
                    "w_src": torch.tensor(w_src, device=device),
                    "w_tgt": torch.tensor(w_tgt, device=device),
                }
                # L_id = torch.tensor(0.0, device=device)
                # if is_id.any():
                #     L_id = F.l1_loss(pred_noise[is_id], noise[is_id])
                # # L_edge = torch.tensor(0.0, device=device)
                # # # if args.use_edge_loss and (~is_id).any():
                # # #      L_edge = compute_edge_loss(src_img[~is_id], tgt_img[~is_id])
                # loss_dict = {"loss": L_diff + args.lambda_id * L_id + L_var}
                

            else:
                if args.boxes_path is None:
                    x, y = data
                    boxes_mask = None
                else:
                    x, y, boxes_mask, _ = data
                boxes_mask = boxes_mask.unsqueeze(1).to(device)

                x = x.to(device)
                y = y.to(device)

                with torch.no_grad():
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)

                t = torch.randint(
                    0,
                    diffusion.num_timesteps,
                    (x.shape[0],),
                    device=device
                )

                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(
                    model,
                    x,
                    t,
                    boxes_mask,
                    args.mask_rl,
                    model_kwargs,
                )

            loss = loss_dict["loss"].mean()
            (loss / accum_steps).backward()
            accum_counter += 1
            if accum_counter % accum_steps == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)

            running_loss += loss.item()

            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                if rank == 0:
                    log.log(
                        {
                            "step": train_steps,
                            "loss": avg_loss,
                        },
                    )
                running_loss = 0
                log_steps = 0
                start_time = time()

            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": extract_task_specific_parameters(model),
                        "args": args,
                        "certain_betas": diffusion.base_diffusion.betas,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if (
                args.training_sample_steps is not None
                and train_steps % args.training_sample_steps == 0
                and train_steps > 0
            ):
                if rank == 0:
                    model.eval()
                    with torch.no_grad():
                        if args.task_type == "image_editing_paired":
                            # Paired mode preview (stage-specific).
                            n = 4
                            diffusion_sample = create_diffusion(
                                str(250),
                                noise_schedule=args.noise_schedule,
                                certain_betas=diffusion.base_diffusion.betas,
                            )

                            if args.train_stage == "stage1":
                                # Stage1 preview: pure noise + weather label.
                                core_model = model.module if hasattr(model, "module") else model
                                if hasattr(core_model, "backbone"):
                                    core_model = core_model.backbone
                                weather_to_idx = PairedWeatherDataset.WEATHER_TO_IDX
                                src_idx = weather_to_idx[args.source_weather]
                                tgt_idx = weather_to_idx[args.target_weather]
                                half_n = n // 2
                                class_labels = torch.tensor(
                                    [src_idx] * half_n + [tgt_idx] * (n - half_n),
                                    device=device,
                                    dtype=torch.long,
                                )
                                z = torch.randn(n, 4, latent_size, latent_size, device=device)
                                model_kwargs = {"y_tgt": class_labels}
                                samples = diffusion_sample.p_sample_loop(
                                    core_model,
                                    z.shape,
                                    z,
                                    clip_denoised=False,
                                    model_kwargs=model_kwargs,
                                    progress=True,
                                    device=device,
                                )
                                samples = vae.decode(samples / 0.18215).sample
                                samples = torch.clamp((samples + 1.0) / 2.0, 0.0, 1.0)
                                source_display = samples
                            else:
                                # Stage2 preview: DDIM inversion sunny -> rain editing.
                                preview_batch = next(iter(loader))
                                src_imgs = preview_batch["src_img"][:n].to(device)
                                src_label = preview_batch["src_label"][:n]
                                tgt_label = preview_batch["tgt_label"][:n]
                                if not isinstance(src_label, torch.Tensor):
                                    src_label = torch.tensor(src_label, device=device).long()
                                else:
                                    src_label = src_label.to(device).long()
                                if not isinstance(tgt_label, torch.Tensor):
                                    tgt_label = torch.tensor(tgt_label, device=device).long()
                                else:
                                    tgt_label = tgt_label.to(device).long()
                                src_seg = seg_extractor(src_imgs)
                                src_edge = sobel_edges(src_imgs)
                                inverter = create_inverter("ddim", diffusion_sample, model, vae, device)
                                edited_pixels, _ = inverter.edit(
                                    img_tensor=src_imgs,
                                    source_weather_label=src_label,
                                    target_weather_label=tgt_label,
                                    src_img=src_imgs,
                                    seg_logits=src_seg,
                                    edge_map=src_edge,
                                    inversion_steps=50,
                                    denoise_steps=50,
                                    cfg_scale=4.0,
                                )
                                samples = edited_pixels / 255.0
                                source_display = torch.clamp((src_imgs + 1.0) / 2.0, 0.0, 1.0)

                        else:
                            # Original generation mode preview.
                            class_labels = [
                                i for i in range(args.scenario_num) for _ in range(4)
                            ]

                            n = len(class_labels)
                            z = torch.randn(n, 4, 32, 32, device=device)
                            y = torch.tensor(class_labels, device=device)

                            z = torch.cat([z, z], 0)
                            y_null = torch.tensor([args.scenario_num] * n, device=device)
                            y = torch.cat([y, y_null], 0)
                            model_kwargs = dict(y=y, cfg_scale=4)
                            diffusion_sample = create_diffusion(
                                str(250),
                                noise_schedule=args.noise_schedule,
                                certain_betas=diffusion.base_diffusion.betas,
                            )

                            samples = diffusion_sample.p_sample_loop(
                                model.module.forward_with_cfg,
                                z.shape,
                                z,
                                clip_denoised=False,
                                model_kwargs=model_kwargs,
                                progress=True,
                                device=device,
                            )
                            samples, _ = samples.chunk(2, dim=0)
                            samples = vae.decode(samples / 0.18215).sample
                            source_display = samples
                        log.log(
                            {
                                "step": train_steps,
                                "source_img": wandb.Image(
                                        source_display.float(),
                                        caption=f"source-step-{train_steps:06d}"
                                    ),
                                "edited_img": wandb.Image(
                                        samples.float(),
                                        caption=f"edited-step-{train_steps:06d}"
                                    ),
                                "loss": loss.item(),
                            },
                        )
                    model.train()
                dist.barrier()

        if accum_counter % accum_steps != 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

    model.eval()
    logger.info("Done!")
    cleanup()


def run():
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--boxes-path", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--model",
        type=str,
        choices=list(DiT_models.keys()),
        default="DiT-XL/2",
    )
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=100)

    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--vae-checkpoint", type=str, default=None)
    parser.add_argument("--modulation-checkpoint", type=str, default=None)
    parser.add_argument("--embed-checkpoint", type=str, default=None)

    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--training_sample_steps", type=int, default=None)
    parser.add_argument("--scenario_num", type=int, default=None)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--modulation", action="store_true")
    parser.add_argument("--patch_modulation", action="store_true")
    parser.add_argument("--block_mlp_modulation", action="store_true")
    parser.add_argument("--cond_mlp_modulation", action="store_true")

    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--finetune_depth", type=int, default=28)
    parser.add_argument("--mask_rl", type=float, default=1)
    parser.add_argument("--noise_schedule", type=str, default="linear")

    #
    parser.add_argument("--task_type", type=str, default="generation",
                       choices=["generation", "image_editing_paired"])
    parser.add_argument("--weather_data_path", type=str, default=None)
    parser.add_argument("--source_weather", type=str, default="sunny")
    parser.add_argument("--target_weather", type=str, default="rain")
    parser.add_argument("--edit_timestep_min", type=int, default=300)
    parser.add_argument("--edit_timestep_max", type=int, default=700)
    parser.add_argument("--use_perceptual_loss", action="store_true")
    parser.add_argument("--use_edge_loss", action="store_true")
     #
    # parser.add_argument("--use_src_cond", action="store_true",
    # parser.add_argument("--lambda_id", type=float, default=1.0,
    #                 help="identity loss ")
    # parser.add_argument("--lambda_edge", type=float, default=0.5,
    #                 help="edge consistency loss ")
    # parser.add_argument("--identity_ratio", type=float, default=0.4,
    #
    parser.add_argument("--lr_dit", type=float, default=1e-5)
    parser.add_argument("--lr_new", type=float, default=1e-4)
    parser.add_argument("--lambda_latent", type=float, default=0.5)
    parser.add_argument("--lambda_app", type=float, default=0.5)
    parser.add_argument("--lambda_edge", type=float, default=0.2)
    parser.add_argument("--lambda_seg", type=float, default=0.5)
    parser.add_argument("--lambda_lpips", type=float, default=0.1)
    parser.add_argument("--lambda_var", type=float, default=0.1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--train_stage", type=str, default="stage1", choices=["stage1", "stage2"])
    parser.add_argument("--w_src_start", type=float, default=0.3)
    parser.add_argument("--w_tgt_start", type=float, default=0.7)
    parser.add_argument("--w_src_end", type=float, default=0.2)
    parser.add_argument("--w_tgt_end", type=float, default=0.8)
    parser.add_argument("--mix_anneal_start_step", type=int, default=30000)
    parser.add_argument("--mix_anneal_end_step", type=int, default=90000)
    parser.add_argument("--stage2_low_t_min", type=int, default=0)
    parser.add_argument("--stage2_low_t_max", type=int, default=300)
    parser.add_argument("--stage2_low_t_prob", type=float, default=0.2)
    args = parser.parse_args()

   
    
    run()
    print(args)
    main(args)

