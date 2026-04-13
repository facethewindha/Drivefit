"""
天气图像编辑推理 (使用 DDIM Inversion)
Usage: python sample_edit.py --input_image sunny.png --target_weather rain
"""
import torch, argparse, os
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL
from drivefit_models import DiT_models
from diffusion import create_diffusion
from inversion import create_inverter


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}

    # 1. 加载模型
    model = DiT_models[args.model](
        input_size=32, num_classes=1000, modulation=True,
        cond_mlp_modulation=True, rank=2, scenario_num=2,
        rope=True, finetune_depth=28,use_src_cond=True) # 推理时 model 创建需要开启 use_src_cond
    if args.pretrained_checkpoint:
        s = torch.load(args.pretrained_checkpoint, map_location="cpu")
        model.load_state_dict(s if not isinstance(s, dict) or "model" not in s
                              else s["model"], strict=False)
    if args.checkpoint:
        s = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(s["model"] if "model" in s else s, strict=False)
    model = model.to(device).eval()

    # 2. 加载 VAE & Diffusion
    vae = AutoencoderKL.from_pretrained(args.vae_checkpoint).to(device)
    diffusion = create_diffusion(str(args.num_sampling_steps),
                                  noise_schedule="linear", flag=0)

    # 3. 创建 Inverter (统一接口)
    inverter = create_inverter(args.inversion_type, diffusion, model, vae, device)

    # 4. 加载输入图像
    transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)])
    input_tensor = transform(Image.open(args.input_image).convert("RGB")).unsqueeze(0).to(device)

    # 5. 编辑 (source 条件用 sunny, target 条件用 rain)
    target_idx = WEATHER_TO_IDX[args.target_weather]
    source_kwargs = dict(y=torch.tensor([0], device=device))        # sunny
    target_kwargs = dict(y=torch.tensor([target_idx], device=device)) # rain

    edited_pixels, _ = inverter.edit(
        input_tensor, target_idx,
        inversion_steps=args.inversion_steps,
        denoise_steps=args.denoise_steps,
        source_model_kwargs=source_kwargs,
        target_model_kwargs=target_kwargs,
        cfg_scale=args.cfg_scale)

    # 6. 保存
    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path,
        f"edited_{args.target_weather}_{os.path.basename(args.input_image)}")
    out_np = edited_pixels.permute(0,2,3,1).cpu().numpy().astype(np.uint8)[0]
    Image.fromarray(out_np).save(out_file)
    print(f"✓ 编辑完成: {out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_image", type=str, required=True)
    p.add_argument("--target_weather", type=str, default="rain",
                   choices=["rain","snow","cloud","night"])
    p.add_argument("--inversion_type", type=str, default="ddim",
                   help="inversion方法: ddim | (后续: null_text, edict)")
    p.add_argument("--inversion_steps", type=int, default=50)
    p.add_argument("--denoise_steps", type=int, default=50)
    p.add_argument("--edit_timestep", type=int, default=500)
    p.add_argument("--output_path", type=str, default="./edited_output")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--pretrained_checkpoint", type=str,
                   default="./pretrained_models/DiT-XL-2-256x256.pt")
    p.add_argument("--vae_checkpoint", type=str,
                   default="./pretrained_models/sd-vae-ft-ema")
    p.add_argument("--model", type=str, default="DiT-XL/2")
    p.add_argument("--cfg_scale", type=float, default=4.0)
    p.add_argument("--num_sampling_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    main(args)
