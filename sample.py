"""
测试脚本：生成图片并计算FID分数
Usage: python sample.py --checkpoint <checkpoint_path>
"""

import torch
import argparse
from drivefit_models import DiT_models
from utils import generate_and_fid

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 加载模型
    print(f"Loading model: {args.model}")
    model = DiT_models[args.model](
        input_size=32,
        num_classes=1000,
        modulation=True,
        cond_mlp_modulation=True,
        rank=2,
        scenario_num=5,
        rope=True,
        finetune_depth=28,
    )
    
    # 2. 生成图片并计算FID
    print(f"Generating {args.num_samples} samples...")
    model_paths = [
        args.vae_checkpoint,
        args.pretrained_checkpoint,
        args.checkpoint
    ]
    
    generate_and_fid(
        model=model,
        model_path=model_paths,
        real_dataset_path=args.real_data_path,
        save_dir=args.output_dir,
        per_proc_sample_num=args.num_samples,
        per_proc_batch_size=args.batch_size,
        num_domain_class=5,
        dm_config={
            "latent_size": 32,
            "num_sampling_steps": 250,
            "noise_schedule": args.noise_schedule,
            "certain_betas": None,
            "flag": 0,
        },
        device=device,
        use_ddp=False,
        load_ckpt=True,
        cfg_scale=4.0,
        seed=args.seed,
    )
    
    print(f"✓ 完成！生成的图片保存在: {args.output_dir}")
    print(f"✓ FID分数已保存")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="训练好的checkpoint路径，例如：results/ithaca365/000-DiT-XL-2/checkpoints/0175000.pt")
    parser.add_argument("--pretrained-checkpoint", type=str, 
                        default="./pretrained_models/DiT-XL-2-256x256.pt",
                        help="预训练模型路径")
    parser.add_argument("--vae-checkpoint", type=str,
                        default="./pretrained_models/sd-vae-ft-ema",
                        help="VAE模型路径")
    parser.add_argument("--real-data-path", type=str,
                        default="./datasets/Ithaca365/Ithaca365-scenario",
                        help="真实数据集路径（用于计算FID）")
    parser.add_argument("--output-dir", type=str,
                        default="./generated_samples",
                        help="生成图片保存目录")
    parser.add_argument("--model", type=str, default="DiT-XL/2",
                        help="模型名称")
    parser.add_argument("--num-samples", type=int, default=1000,
                        help="生成图片数量（建议1000-5000）")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="生成时的batch size")
    parser.add_argument("--noise-schedule", type=str, default="progress",
                        help="噪声调度策略")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子")
    
    args = parser.parse_args()
    main(args)
