"""Weather image editing inference with DDIM inversion (paired editing path)."""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from diffusers.models import AutoencoderKL

from diffusion import create_diffusion
from drivefit_models import DiT_models
from editing.conditioning import WeatherEditWrapper
from editing.edge_ops import sobel_edges
from editing.seg_extractor import FrozenSegFormer
from inversion import create_inverter


WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}


def init_weather_embed_from_ssei(model, embed_ckpt_path, source_weather="sunny", target_weather="rain"):
    if embed_ckpt_path is None:
        return
    ckpt = torch.load(embed_ckpt_path, map_location="cpu")
    key = "y_embedder.scenario_embedding_table.weight"
    if key not in ckpt:
        return

    w = ckpt[key]
    src_idx = WEATHER_TO_IDX[source_weather]
    tgt_idx = WEATHER_TO_IDX[target_weather]
    null_idx = w.shape[0] - 1
    if src_idx >= null_idx or tgt_idx >= null_idx:
        return

    selected = torch.stack([w[src_idx], w[tgt_idx], w[null_idx]], dim=0).clone()

    m = model.backbone if hasattr(model, "backbone") else model
    if not hasattr(m, "src_weather_embedder") or not hasattr(m, "tgt_weather_embedder"):
        return

    with torch.no_grad():
        if m.src_weather_embedder.scenario_embedding_table.weight.shape == selected.shape:
            m.src_weather_embedder.scenario_embedding_table.weight.copy_(selected)
        if m.tgt_weather_embedder.scenario_embedding_table.weight.shape == selected.shape:
            m.tgt_weather_embedder.scenario_embedding_table.weight.copy_(selected)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = DiT_models[args.model](
        input_size=32,
        num_classes=1000,
        modulation=True,
        cond_mlp_modulation=True,
        rank=args.rank,
        scenario_num=2,
        rope=True,
        finetune_depth=28,
        use_src_cond=False,
    )

    if args.pretrained_checkpoint:
        s = torch.load(args.pretrained_checkpoint, map_location="cpu")
        state = s["model"] if isinstance(s, dict) and "model" in s else s
        base_model.load_state_dict(state, strict=False)

    model = WeatherEditWrapper(
        backbone=base_model,
        seg_channels=19,
        d_model=base_model.pos_embed.shape[-1],
        num_heads=base_model.num_heads,
    )

    if args.embed_checkpoint:
        init_weather_embed_from_ssei(model, args.embed_checkpoint, args.source_weather, args.target_weather)

    if args.checkpoint:
        s = torch.load(args.checkpoint, map_location="cpu")
        state = s["model"] if isinstance(s, dict) and "model" in s else s
        model.load_state_dict(state, strict=False)

    model = model.to(device).eval()

    vae = AutoencoderKL.from_pretrained(args.vae_checkpoint).to(device)
    seg_extractor = FrozenSegFormer(
        model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        out_size=(256, 256),
    ).to(device)

    diffusion = create_diffusion(
        str(args.num_sampling_steps),
        noise_schedule="linear",
        flag=0,
    )

    inverter = create_inverter(args.inversion_type, diffusion, model, vae, device)

    tfm = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    src_img = tfm(Image.open(args.input_image).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_src = seg_extractor(src_img)
        edge_src = sobel_edges(src_img)

    src_idx = WEATHER_TO_IDX[args.source_weather]
    tgt_idx = WEATHER_TO_IDX[args.target_weather]

    edited_pixels, _ = inverter.edit(
        img_tensor=src_img,
        source_weather_label=src_idx,
        target_weather_label=tgt_idx,
        src_img=src_img,
        seg_logits=seg_src,
        edge_map=edge_src,
        inversion_steps=args.inversion_steps,
        denoise_steps=args.denoise_steps,
        cfg_scale=args.cfg_scale,
    )

    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, f"edited_{args.target_weather}_{os.path.basename(args.input_image)}")
    out_np = edited_pixels.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    Image.fromarray(out_np).save(out_file)
    print(f"saved: {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_image", type=str, required=True)
    p.add_argument("--source_weather", type=str, default="sunny", choices=["sunny", "rain", "snow", "cloud", "night"])
    p.add_argument("--target_weather", type=str, default="rain", choices=["rain", "snow", "cloud", "night", "sunny"])
    p.add_argument("--inversion_type", type=str, default="ddim")
    p.add_argument("--inversion_steps", type=int, default=50)
    p.add_argument("--denoise_steps", type=int, default=50)
    p.add_argument("--output_path", type=str, default="./edited_output")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--pretrained_checkpoint", type=str, default="./pretrained_models/DiT-XL-2-256x256.pt")
    p.add_argument("--embed_checkpoint", type=str, default="./pretrained_models/clip_similarity_embed.pt")
    p.add_argument("--vae_checkpoint", type=str, default="./pretrained_models/sd-vae-ft-ema")
    p.add_argument("--model", type=str, default="DiT-XL/2")
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--cfg_scale", type=float, default=3.0)
    p.add_argument("--num_sampling_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    main(args)
