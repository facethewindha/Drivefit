import torch
import sys
sys.path.append(".")
from drivefit_models import DiT_models

model = DiT_models["DiT-XL/2"](
    input_size=32,
    num_classes=1000,
    modulation=True,
    cond_mlp_modulation=True,
    rank=4,
    scenario_num=5,
    rope=True,
    finetune_depth=28,
    use_bbox_cond=False,
    max_boxes=30,
)

# 加载预训练模型（设置 requires_grad）
from train import requires_grad
requires_grad(model, True, 28)

# 查看可训练参数
print("=" * 60)
print("可训练参数:")
print("=" * 60)
trainable = 0
frozen = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  ✅ {name:60s} {list(param.shape)}")
        trainable += param.numel()
    else:
        frozen += param.numel()

print("=" * 60)
print(f"可训练参数: {trainable:,}")
print(f"冻结参数:   {frozen:,}")
print(f"总参数:     {trainable + frozen:,}")
print(f"可训练比例: {trainable / (trainable + frozen) * 100:.2f}%")