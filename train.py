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

os.environ["NCCL_SOCKET_IFNAME"] = "enp0s31f6"
os.environ["NCCL_IB_DISABLE"] = "1"

from drivefit_models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from dataset import CustomImageFolder
from inversion import create_inverter

import wandb

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
    if model.modulation:
        for name, param in model.named_parameters():
            if flag:
                if (
                    name.find("weight_modulation") >= 0
                    or name.find("scenario_embedding_table") >= 0
                ):
                    if name.find("blocks") >= 0:
                        param.requires_grad = int(name.split(".")[1]) < depth
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
    else:
        for name, param in model.named_parameters():
            param.requires_grad = flag


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
#新增辅助函数
def compute_edge_loss(img1, img2):
    """Sobel 边缘一致性 loss，img1/img2 均为 (B,3,H,W)，值域 [-1,1]"""
    import torch.nn.functional as F
    # Sobel 核，检测水平+垂直边缘
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=img1.dtype, device=img1.device).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    # 转灰度
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
    
    # 在logger创建后加载checkpoint，确保日志被记录
    resume_epoch = 0
    resume_train_steps = 0
    
    logger.info(f"[DEBUG] args.resume_checkpoint = {args.resume_checkpoint}")

    # Create model:
    assert (
        args.image_size % 8 == 0
    ), "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        modulation=args.modulation,
        patch_modulation=args.patch_modulation,
        block_mlp_modulation=args.block_mlp_modulation,
        cond_mlp_modulation=args.cond_mlp_modulation,
        rank=args.rank,
        #新增参数
        use_src_cond=(args.task_type == "image_editing"),
        # 修改参数
        # scenario_num=args.scenario_num if args.dataset_name is not None else 0,
        scenario_num=2 if args.task_type == "image_editing" else 
        (args.scenario_num if args.dataset_name is not None else 0),

        rope=args.rope,
        finetune_depth=args.finetune_depth,
    )

    # 1. 首先加载预训练模型（基础权重，691M参数）
    if args.resume_checkpoint is not None:
        logger.info(f"Loading pretrained model from: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
        state_dict = checkpoint["model"] if (isinstance(checkpoint, dict) and "model" in checkpoint) else checkpoint
        # --- 新增：处理 8 通道权重的缝合逻辑 ---
        if args.use_src_cond and 'x_embedder.proj.weight' in state_dict:
             # 原始权重是 [hidden, 4, p, p]
             old_weight = state_dict['x_embedder.proj.weight']
             # 检查当前模型的权重形状 [hidden, 8, p, p]
             # 我们通过 model 对象直接获取目标形状
             target_shape = model.module.x_embedder.proj.weight.shape if hasattr(model, "module") \
                           else model.x_embedder.proj.weight.shape
            
             if old_weight.shape[1] == 4 and target_shape[1] == 8:
                # 创建全 0 的 8 通道新权重
                new_weight = torch.zeros(target_shape)
                # 前 4 通道承接预训练噪声模型的生成能力
                new_weight[:, :4, :, :] = old_weight
                new_weight[:, 4:, :, :] = old_weight # 通道克隆
                # 后 4 通道（对应 z_src）初始化为 0
                state_dict['x_embedder.proj.weight'] = new_weight
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            # 如果是训练保存的格式，提取model部分
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            # 预训练模型（直接是 state_dict）
            model.load_state_dict(checkpoint, strict=False)
        logger.info("Loaded pretrained base model")

    # 2. 然后加载微调参数（353个参数）
    if args.modulation_checkpoint is not None:
        args.modulation_checkpoint = args.modulation_checkpoint.strip()
        logger.info(f"Loading fine-tuned checkpoint from: {args.modulation_checkpoint}")
        checkpoint = torch.load(args.modulation_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        resume_epoch = checkpoint.get("epoch", 0)
        resume_train_steps = checkpoint.get("train_steps", 0)
        logger.info(f"Loaded fine-tuned params, resuming from epoch {resume_epoch}, step {resume_train_steps}")

    # 3. 加载embed checkpoint
    #如果是图片编辑并且有编码器检查点，才会加载编码器
    # if args.embed_checkpoint is not None:
    if args.embed_checkpoint is not None and args.task_type != "image_editing":
        checkpoint = torch.load(args.embed_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)

    requires_grad(model, True, args.finetune_depth)

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

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    # 图片转换
    transform = transforms.Compose( 
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    #修改数据加载分支，如果是图片编辑任务，使用WeatherEditDataset，否则使用ImageFolder
    if args.task_type == "image_editing":
        from dataset import WeatherEditDataset
        dataset = WeatherEditDataset(
            args.weather_data_path, args.source_weather,
            args.target_weather, transform=transform,split="train", split_ratio=0.8,identity_ratio=args.identity_ratio)
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

        diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=args.noise_schedule,
            flag=epoch // interval,
        )
        # 修改data分支，如果是编辑，同步修改的dataset
        for data in loader:
            if args.task_type == "image_editing":
                src_img  = data["src_img"].to(device)
                tgt_img  = data["tgt_img"].to(device)
                y_src    = torch.tensor(data["src_label"], device=device).long() \
                           if not isinstance(data["src_label"], torch.Tensor) \
                           else data["src_label"].to(device).long()
                y_tgt    = torch.tensor(data["tgt_label"], device=device).long() \
                           if not isinstance(data["tgt_label"], torch.Tensor) \
                           else data["tgt_label"].to(device).long()
                is_id    = data["is_identity"].bool().to(device)

                with torch.no_grad():
                    z_src = vae.encode(src_img).latent_dist.sample().mul_(0.18215)
                    z_tgt = vae.encode(tgt_img).latent_dist.sample().mul_(0.18215)

                t = torch.randint(
                    args.edit_timestep_min,
                    args.edit_timestep_max,
                    (z_tgt.shape[0],),
                    device=device
                )
                # 拼接源图 latent 作为 channel 条件
                # 训练时的 x 是 noisy target，但模型 forward 需要拿到 z_src
                # 通过 model_kwargs 传递 z_src，在 training_losses 之外拼接
                # 1. 准备输入 x（拼接 z_tgt_noisy 和 z_src）
                noise = torch.randn_like(z_tgt)
                z_tgt_noisy = diffusion.q_sample(z_tgt, t, noise=noise)
                x_input = torch.cat([z_tgt_noisy, z_src], dim=1)  # (B, 8, 32, 32)
                model_kwargs = dict(y_src=y_src, y_tgt=y_tgt)
                model_out = model(x_input, t, **model_kwargs)
                #增加方差损失
                # 1. 拆分输出
                C = z_tgt.shape[1] # C 为 4
                pred_noise, model_var_values = torch.split(model_out, C, dim=1)
                # 2. 计算预测噪声与真实噪声的 MSE
                L_diff = F.mse_loss(pred_noise, noise)
                # 3. 计算方差损失 (VB Loss)
                # 只有在 learn_sigma=True 时才需要
                L_var = torch.tensor(0.0, device=device)
                if (model.module if hasattr(model, "module") else model).learn_sigma:
                    # 目标方差
                    # 使用 diffusion 内部的辅助函数计算 Variational Bound
                    # 冻结均值预测，只训练方差分支
                    frozen_out = torch.cat([pred_noise.detach(), model_var_values], dim=1)
                    vb_terms = diffusion._vb_terms_bpd(
                        model=lambda *args, r=frozen_out: r,
                        x_start=z_tgt,
                        x_t=z_tgt_noisy, # 注意：这里用的是 4 通道的加噪图
                        t=t,
                        clip_denoised=False,
                    )
                    L_var = vb_terms["output"]
                    if diffusion.loss_type.name == "RESCALED_MSE":
                        L_var *= diffusion.num_timesteps / 1000.0
                # Identity loss（仅对 is_identity=True 的样本）
                L_id = torch.tensor(0.0, device=device)
                if is_id.any():
                    L_id = F.l1_loss(pred_noise[is_id], noise[is_id])
                # Edge loss（仅对 editing 样本）
                # L_edge = torch.tensor(0.0, device=device)
                # # if args.use_edge_loss and (~is_id).any():
                # #      L_edge = compute_edge_loss(src_img[~is_id], tgt_img[~is_id])
                loss_dict = {"loss": L_diff + args.lambda_id * L_id + L_var}
                

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
            opt.zero_grad()
            loss.backward()
            opt.step()

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

            if train_steps % args.training_sample_steps == 0 and train_steps > 0:
                if rank == 0:
                    model.eval()
                    with torch.no_grad():
                        if args.task_type == "image_editing":
                            # 编辑模式：从训练集取一批source图，做前向去噪预览
                            scenario_num = 2  # sunny/rain 两个类别
                            n = 4  # 预览4张图
                            # 取训练集前几张作为输入
                            preview_batch = next(iter(loader))
                            src_imgs = preview_batch["src_img"][:n].to(device)
                            src_label = preview_batch["src_label"][0].item()
                            # 假设 batch 内 src_label 相同
                            tgt_label = preview_batch["tgt_label"][0].item()
                            # if not isinstance(tgt_label, torch.Tensor):
                            #     tgt_label = torch.tensor([tgt_label] * n)
                            # tgt_label = tgt_label.to(device)

                            x = vae.encode(src_imgs).latent_dist.sample().mul_(0.18215)

                            # 加噪到 edit_timestep_max，再去噪回来
                            diffusion_sample = create_diffusion(
                                str(250),
                                noise_schedule=args.noise_schedule,
                                certain_betas=diffusion.base_diffusion.betas,
                            )
                            # source_kwargs = dict(y=torch.tensor([0] * n, device=device))  # sunny
                            # target_kwargs = dict(y=torch.tensor([1] * n, device=device))  # rain
                            inverter = create_inverter("ddim", diffusion_sample, model, vae, device)
                            edited_pixels, _ = inverter.edit(
                                src_imgs, # 现在可以传入 (n, 3, H, W)
                                src_label,
                                tgt_label,
                                inversion_steps=50,
                                denoise_steps=50,
                                cfg_scale=4.0
                            )
                            samples = edited_pixels / 255.0  # wandb.Image 需要 [0,1]

                        else:
                            # 原始生成模式
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
                        
                        
                        
                        source_display = (src_imgs + 1.0) / 2.0
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

    model.eval()
    logger.info("Done!")
    cleanup()


def run():
    # 从当前进程的环境变量里取出几项分布式训练常用配置，并打印出来，方便排查进程组初始化问题
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

    #新增参数
    parser.add_argument("--task_type", type=str, default="generation",
                       choices=["generation", "image_editing"])
    parser.add_argument("--weather_data_path", type=str, default=None)
    parser.add_argument("--source_weather", type=str, default="sunny")
    parser.add_argument("--target_weather", type=str, default="rain")
    parser.add_argument("--edit_timestep_min", type=int, default=300)
    parser.add_argument("--edit_timestep_max", type=int, default=700)
    parser.add_argument("--use_perceptual_loss", action="store_true")
    parser.add_argument("--use_edge_loss", action="store_true")
     #新增参数
    parser.add_argument("--use_src_cond", action="store_true",
                    help="开启 channel concat 源图条件")
    parser.add_argument("--lambda_id", type=float, default=1.0,
                    help="identity loss 权重")
    parser.add_argument("--lambda_edge", type=float, default=0.5,
                    help="edge consistency loss 权重")
    parser.add_argument("--identity_ratio", type=float, default=0.4,
                    help="每个 batch 中 identity 样本的比例")
    args = parser.parse_args()
   
    
    run()
    print(args)
    main(args)
