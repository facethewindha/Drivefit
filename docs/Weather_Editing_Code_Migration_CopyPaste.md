# Weather Editing 代码迁移方案（可直接复制粘贴）

> 说明
- 本文档只提供迁移代码，不改仓库现有实现。
- 每个改动都给出 `原代码` 与 `改后代码`。
- 你可按文件逐段复制。

---

## 1) `dataset.py`

## 1.1 原代码（当前训练数据类）

```python
class WeatherEditDataset(Dataset):
    """
    非配对天气编辑数据集。数据组织:
        data_root/sunny/xxx.png
        data_root/rain/xxx.png
    返回: src_img, tgt_img, src_label, tgt_label, is_identity
    """
    WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}

    def __init__(self, data_root, source_weather="sunny", target_weather="rain",
                 transform=None, identity_ratio=0.4,
                 split="train", split_ratio=0.8, seed=42):
        ...

    def __getitem__(self, index):
        ...
        return {
            "src_img": src_img,
            "tgt_img": tgt_img,
            "src_label": src_label,
            "tgt_label": tgt_label,
            "is_identity": int(is_identity),
        }
```

## 1.2 改后代码（新增 paired 数据类，训练改用它）

> 直接把下面整段粘贴到 `dataset.py`（建议放在 `WeatherEditDataset` 后面）。

```python
import json


class PairedWeatherDataset(Dataset):
    """
    Paired 数据集，样本必须一一对应：
        data_root/sunny/<id>.png
        data_root/rain/<id>.png

    split_file: 每行一个样本 id（不带后缀）
    返回:
        src_img, tgt_img, src_label, tgt_label, sample_id
    """
    WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}

    def __init__(
        self,
        data_root,
        split_file,
        source_weather="sunny",
        target_weather="rain",
        transform=None,
        exts=(".png", ".jpg", ".jpeg"),
    ):
        super().__init__()
        self.data_root = data_root
        self.source_weather = source_weather
        self.target_weather = target_weather
        self.transform = transform
        self.src_label = self.WEATHER_TO_IDX[source_weather]
        self.tgt_label = self.WEATHER_TO_IDX[target_weather]
        self.exts = exts

        with open(split_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]

        self.samples = []
        for sid in ids:
            src_path = self._resolve_image(os.path.join(data_root, source_weather), sid)
            tgt_path = self._resolve_image(os.path.join(data_root, target_weather), sid)
            if src_path is None or tgt_path is None:
                continue
            self.samples.append((sid, src_path, tgt_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid paired samples found from split file: {split_file}")

    def _resolve_image(self, folder, stem):
        for ext in self.exts:
            p = os.path.join(folder, stem + ext)
            if os.path.isfile(p):
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        from PIL import Image as PILImage

        sid, src_path, tgt_path = self.samples[index]
        src_img = PILImage.open(src_path).convert("RGB")
        tgt_img = PILImage.open(tgt_path).convert("RGB")

        if self.transform is not None:
            src_img = self.transform(src_img)
            tgt_img = self.transform(tgt_img)

        return {
            "src_img": src_img,
            "tgt_img": tgt_img,
            "src_label": self.src_label,
            "tgt_label": self.tgt_label,
            "sample_id": sid,
        }
```

---

## 2) 新增文件 `scripts/make_paired_split.py`

> 新建该文件，用于把 1225 对划分成 980/122/123。

```python
import os
import random
import argparse


def list_stems(folder):
    stems = set()
    for fn in os.listdir(folder):
        if fn.lower().endswith((".png", ".jpg", ".jpeg")):
            stems.add(os.path.splitext(fn)[0])
    return stems


def write_lines(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(x + "\n")


def main(args):
    sunny_dir = os.path.join(args.data_root, "sunny")
    rain_dir = os.path.join(args.data_root, "rain")

    sunny = list_stems(sunny_dir)
    rain = list_stems(rain_dir)
    paired = sorted(list(sunny.intersection(rain)))

    if len(paired) < 1225:
        print(f"[WARN] paired count = {len(paired)}, less than 1225")

    random.seed(args.seed)
    random.shuffle(paired)

    n_train, n_val, n_test = args.train, args.val, args.test
    assert n_train + n_val + n_test <= len(paired)

    train_ids = paired[:n_train]
    val_ids = paired[n_train:n_train + n_val]
    test_ids = paired[n_train + n_val:n_train + n_val + n_test]

    write_lines(os.path.join(args.out_dir, "sunny_rain_train.txt"), train_ids)
    write_lines(os.path.join(args.out_dir, "sunny_rain_val.txt"), val_ids)
    write_lines(os.path.join(args.out_dir, "sunny_rain_test.txt"), test_ids)

    print(f"paired total: {len(paired)}")
    print(f"train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="datasets/splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train", type=int, default=980)
    p.add_argument("--val", type=int, default=122)
    p.add_argument("--test", type=int, default=123)
    args = p.parse_args()
    main(args)
```

---

## 3) 新增文件 `editing/conditioning.py`

> 新建目录 `editing`，再新建该文件。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvTokenEncoder(nn.Module):
    """
    输入 BxCxHxW -> 输出 BxNxd_model token
    """
    def __init__(self, in_ch, d_model=1152, base_ch=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 2),
            nn.SiLU(),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_ch * 4),
            nn.SiLU(),
            nn.Conv2d(base_ch * 4, d_model, 1),
        )

    def forward(self, x):
        feat = self.net(x)                # B, d_model, h, w
        b, c, h, w = feat.shape
        tok = feat.flatten(2).transpose(1, 2)  # B, h*w, d_model
        return tok


class CrossAttentionAdapter(nn.Module):
    def __init__(self, d_model=1152, num_heads=16, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, ctx):
        q = self.norm_q(x)
        kv = self.norm_kv(ctx)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return x + self.proj(out)


class WeatherEditWrapper(nn.Module):
    """
    包装已有 DiT backbone:
    - 主输入: noisy sunny latent (4ch)
    - 条件: src_img / seg_logits / edge_map
    - 注入: 在指定 block 做 cross-attn
    """
    def __init__(
        self,
        backbone,
        seg_channels=19,
        d_model=1152,
        num_heads=16,
        inject_layers=None,
    ):
        super().__init__()
        self.backbone = backbone
        self.src_encoder = ConvTokenEncoder(in_ch=3, d_model=d_model)
        self.str_encoder = ConvTokenEncoder(in_ch=seg_channels + 1, d_model=d_model)
        self.ctx_fuse = nn.Linear(d_model, d_model)

        if inject_layers is None:
            # 默认后 1/3 层注入
            depth = len(self.backbone.blocks)
            inject_layers = list(range(depth * 2 // 3, depth))
        self.inject_layers = set(inject_layers)

        self.adapters = nn.ModuleDict()
        for i in self.inject_layers:
            self.adapters[str(i)] = CrossAttentionAdapter(
                d_model=d_model,
                num_heads=num_heads,
            )

    def build_context_tokens(self, src_img, seg_logits, edge_map):
        src_tok = self.src_encoder(src_img)
        str_in = torch.cat([seg_logits, edge_map], dim=1)
        str_tok = self.str_encoder(str_in)
        ctx = torch.cat([src_tok, str_tok], dim=1)
        ctx = self.ctx_fuse(ctx)
        return ctx

    def forward(self, x, t, y_tgt, src_img, seg_logits, edge_map):
        # 1) token embed
        h = self.backbone.x_embedder(x) + self.backbone.pos_embed

        # 2) condition embedding
        c = self.backbone.t_embedder(t)
        if y_tgt is not None and getattr(self.backbone, "scenario_num", 0) > 0:
            c = c + self.backbone.tgt_weather_embedder(y_tgt, self.training)

        # 3) build context
        ctx = self.build_context_tokens(src_img, seg_logits, edge_map)

        # 4) DiT blocks + optional cross-attn injection
        for i, block in enumerate(self.backbone.blocks):
            h = block(h, c)
            if i in self.inject_layers:
                h = self.adapters[str(i)](h, ctx)

        # 5) output head
        out = self.backbone.final_layer(h, c)
        out = self.backbone.unpatchify(out)
        return out
```

---

## 4) 新增文件 `editing/seg_extractor.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class FrozenSegFormer(nn.Module):
    """
    冻结语义分割先验提取器。
    输入: x in [-1, 1], shape Bx3xHxW
    输出: seg logits, shape BxCxhxw（默认会 resize 到输入大小）
    """
    def __init__(
        self,
        model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        out_size=(256, 256),
    ):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.out_size = out_size

        # ImageNet normalization for segformer
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        # x: [-1,1] -> [0,1]
        x01 = (x + 1.0) / 2.0
        x01 = x01.clamp(0.0, 1.0)
        x_norm = (x01 - self.mean) / self.std

        out = self.model(pixel_values=x_norm)
        logits = out.logits  # B,C,h,w
        logits = F.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=False)
        return logits
```

---

## 5) 新增文件 `editing/edge_ops.py`

```python
import torch
import torch.nn.functional as F


def sobel_edges(x):
    """
    x: Bx3xHxW, range [-1,1]
    return: Bx1xHxW
    """
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    ky = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return edge
```

---

## 6) `train.py`（核心迁移）

## 6.1 原代码（关键差异）

```python
model = DiT_models[args.model](
    ...
    use_src_cond=(args.task_type == "image_editing"),
    scenario_num=2 if args.task_type == "image_editing" else ...,
)

...
if args.task_type == "image_editing":
    from dataset import WeatherEditDataset
    dataset = WeatherEditDataset(..., identity_ratio=args.identity_ratio)

...
if args.task_type == "image_editing":
    z_src = vae.encode(src_img)...
    z_tgt = vae.encode(tgt_img)...
    z_tgt_noisy = diffusion.q_sample(z_tgt, t, noise=noise)
    x_input = torch.cat([z_tgt_noisy, z_src], dim=1)
    model_out = model(x_input, t, y_src=y_src, y_tgt=y_tgt)
    loss = L_diff + lambda_id * L_id + L_var
```

## 6.2 改后代码（可直接粘贴）

### A) 替换 import 区域（在 `train.py` 顶部）

```python
import lpips
from dataset import CustomImageFolder, PairedWeatherDataset
from editing.conditioning import WeatherEditWrapper
from editing.seg_extractor import FrozenSegFormer
from editing.edge_ops import sobel_edges
```

### B) 在 `main(args)` 中，替换 model 构建段

```python
base_model = DiT_models[args.model](
    input_size=latent_size,
    num_classes=args.num_classes,
    modulation=args.modulation,
    patch_modulation=args.patch_modulation,
    block_mlp_modulation=args.block_mlp_modulation,
    cond_mlp_modulation=args.cond_mlp_modulation,
    rank=args.rank,
    use_src_cond=False,  # 迁移后固定 4 通道主输入
    scenario_num=2 if args.task_type == "image_editing_paired" else (args.scenario_num if args.dataset_name is not None else 0),
    rope=args.rope,
    finetune_depth=args.finetune_depth,
)

# 先加载预训练权重到 base_model
if args.resume_checkpoint is not None:
    logger.info(f"Loading pretrained model from: {args.resume_checkpoint}")
    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
    state_dict = checkpoint["model"] if (isinstance(checkpoint, dict) and "model" in checkpoint) else checkpoint
    # 不再做 4->8 权重拼接
    base_model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded pretrained base model")

# 编辑任务使用 wrapper
if args.task_type == "image_editing_paired":
    model = WeatherEditWrapper(
        backbone=base_model,
        seg_channels=19,
        d_model=base_model.pos_embed.shape[-1],
        num_heads=base_model.num_heads,
    )
else:
    model = base_model
```

### C) 替换数据集构建段

```python
if args.task_type == "image_editing_paired":
    split_file = os.path.join(args.split_dir, "sunny_rain_train.txt")
    dataset = PairedWeatherDataset(
        data_root=args.weather_data_path,
        split_file=split_file,
        source_weather=args.source_weather,
        target_weather=args.target_weather,
        transform=transform,
    )
else:
    if args.boxes_path is None:
        dataset = ImageFolder(args.data_path, transform=transform)
    else:
        dataset = CustomImageFolder(args.data_path, args.boxes_path, trans_flip=True, transform=transform)
```

### D) 在 DDP 之前新增冻结模块与损失

```python
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
```

### E) 替换优化器为分组学习率

```python
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
```

### F) 替换训练循环中编辑分支

```python
if args.task_type == "image_editing_paired":
    src_img = data["src_img"].to(device)
    tgt_img = data["tgt_img"].to(device)

    y_tgt = torch.tensor(data["tgt_label"], device=device).long() \
        if not isinstance(data["tgt_label"], torch.Tensor) \
        else data["tgt_label"].to(device).long()

    with torch.no_grad():
        z_s = vae.encode(src_img).latent_dist.sample().mul_(0.18215)
        z_r = vae.encode(tgt_img).latent_dist.sample().mul_(0.18215)

    t = torch.randint(args.t_min, args.t_max, (z_s.shape[0],), device=device)
    noise = torch.randn_like(z_s)
    z_s_noisy = diffusion.q_sample(z_s, t, noise=noise)

    with torch.no_grad():
        seg_src = seg_extractor(src_img)            # Bx19xHxW
    edge_src = sobel_edges(src_img)                 # Bx1xHxW

    model_out = model(
        z_s_noisy,
        t,
        y_tgt=y_tgt,
        src_img=src_img,
        seg_logits=seg_src,
        edge_map=edge_src,
    )

    C = z_s.shape[1]
    pred_noise, model_var_values = torch.split(model_out, C, dim=1)

    # core diffusion loss
    L_diff = F.mse_loss(pred_noise, noise)

    # x0 recon in latent
    z0_hat = diffusion._predict_xstart_from_eps(z_s_noisy, t, pred_noise)
    L_latent = F.l1_loss(z0_hat, z_r)

    # decode for image-level losses
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

    # two-stage schedule
    if args.train_stage == "stage1":
        loss = L_diff + args.lambda_latent * L_latent
    else:
        loss = (
            L_diff
            + args.lambda_latent * L_latent
            + args.lambda_app * L_app
            + args.lambda_edge * L_edge
            + args.lambda_seg * L_seg
        )

    loss_dict = {
        "loss": loss,
        "L_diff": L_diff.detach(),
        "L_latent": L_latent.detach(),
        "L_app": L_app.detach(),
        "L_edge": L_edge.detach(),
        "L_seg": L_seg.detach(),
    }
```

### G) 参数区新增（`argparse`）

```python
parser.add_argument("--task_type", type=str, default="generation", choices=["generation", "image_editing", "image_editing_paired"])
parser.add_argument("--split_dir", type=str, default="datasets/splits")
parser.add_argument("--train_stage", type=str, default="stage1", choices=["stage1", "stage2"])
parser.add_argument("--t_min", type=int, default=200)
parser.add_argument("--t_max", type=int, default=700)

parser.add_argument("--lr_dit", type=float, default=1e-5)
parser.add_argument("--lr_new", type=float, default=1e-4)

parser.add_argument("--lambda_latent", type=float, default=0.5)
parser.add_argument("--lambda_app", type=float, default=0.5)
parser.add_argument("--lambda_edge", type=float, default=0.2)
parser.add_argument("--lambda_seg", type=float, default=0.5)
parser.add_argument("--lambda_lpips", type=float, default=0.1)
```

---

## 7) `sample_edit.py`（推理对齐训练）

## 7.1 原代码（关键问题）

```python
model = DiT_models[...](..., use_src_cond=True)
...
source_kwargs = dict(y=torch.tensor([0], device=device))
target_kwargs = dict(y=torch.tensor([target_idx], device=device))
```

## 7.2 改后代码（整文件替换）

> 直接用下面内容覆盖 `sample_edit.py`。

```python
"""
Paired weather editing inference (sunny -> rain) with context conditioning.
"""
import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers.models import AutoencoderKL

from drivefit_models import DiT_models
from editing.conditioning import WeatherEditWrapper
from editing.seg_extractor import FrozenSegFormer
from editing.edge_ops import sobel_edges
from diffusion import create_diffusion


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}

    # build base model + wrapper
    base_model = DiT_models[args.model](
        input_size=32,
        num_classes=1000,
        modulation=True,
        cond_mlp_modulation=True,
        rank=2,
        scenario_num=2,
        rope=True,
        finetune_depth=28,
        use_src_cond=False,
    )

    if args.pretrained_checkpoint:
        s = torch.load(args.pretrained_checkpoint, map_location="cpu")
        base_model.load_state_dict(s if not isinstance(s, dict) or "model" not in s else s["model"], strict=False)

    model = WeatherEditWrapper(
        backbone=base_model,
        seg_channels=19,
        d_model=base_model.pos_embed.shape[-1],
        num_heads=base_model.num_heads,
    )

    if args.checkpoint:
        s = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(s["model"] if "model" in s else s, strict=False)

    model = model.to(device).eval()

    vae = AutoencoderKL.from_pretrained(args.vae_checkpoint).to(device)
    seg_extractor = FrozenSegFormer(
        model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        out_size=(256, 256),
    ).to(device)

    diffusion = create_diffusion(str(args.num_sampling_steps), noise_schedule="linear", flag=0)

    tfm = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    src_img = tfm(Image.open(args.input_image).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        z_s = vae.encode(src_img).latent_dist.sample().mul_(0.18215)
        seg_src = seg_extractor(src_img)
        edge_src = sobel_edges(src_img)

    y_tgt = torch.tensor([WEATHER_TO_IDX[args.target_weather]], device=device).long()

    # noising to edit strength
    t_edit = torch.tensor([args.t_edit], device=device).long()
    eps = torch.randn_like(z_s)
    z_t = diffusion.q_sample(z_s, t_edit, noise=eps)

    # simple DDPM reverse loop
    img = z_t
    with torch.no_grad():
        for i in reversed(range(args.t_edit)):
            t = torch.tensor([i], device=device).long()
            model_out = model(
                img,
                t,
                y_tgt=y_tgt,
                src_img=src_img,
                seg_logits=seg_src,
                edge_map=edge_src,
            )
            C = img.shape[1]
            pred_eps = model_out[:, :C]
            out = diffusion.p_sample(
                model=lambda x, tt, **kwargs: torch.cat([pred_eps, torch.zeros_like(pred_eps)], dim=1),
                x=img,
                t=t,
                clip_denoised=False,
                model_kwargs={},
            )
            img = out["sample"]

        x_hat = vae.decode(img / 0.18215).sample
        x_hat = torch.clamp((x_hat + 1.0) / 2.0, 0.0, 1.0)

    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, f"edited_{args.target_weather}_{os.path.basename(args.input_image)}")
    out_np = (x_hat[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(out_np).save(out_file)
    print(f"saved: {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_image", type=str, required=True)
    p.add_argument("--target_weather", type=str, default="rain", choices=["rain", "snow", "cloud", "night"])
    p.add_argument("--output_path", type=str, default="./edited_output")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--pretrained_checkpoint", type=str, default="./pretrained_models/DiT-XL-2-256x256.pt")
    p.add_argument("--vae_checkpoint", type=str, default="./pretrained_models/sd-vae-ft-ema")
    p.add_argument("--model", type=str, default="DiT-XL/2")
    p.add_argument("--num_sampling_steps", type=int, default=250)
    p.add_argument("--t_edit", type=int, default=350)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    main(args)
```

---

## 8) `requirements.txt` 追加依赖

## 8.1 原代码
无 `lpips`、`transformers`。

## 8.2 改后代码（追加）

```txt
lpips>=0.1.4
transformers>=4.40.0
opencv-python>=4.8.0
```

---

## 9) `sh/train.sh`（编辑任务命令）

## 9.1 改后完整命令（示例）

```powershell
$env:CUDA_VISIBLE_DEVICES="0"
python train.py --model DiT-XL/2 `
  --task_type image_editing_paired `
  --weather_data_path ./datasets/paired_weather `
  --split_dir ./datasets/splits `
  --source_weather sunny --target_weather rain `
  --epochs 300 --global-batch-size 4 --num_workers 4 `
  --resume-checkpoint ./pretrained_models/DiT-XL-2-256x256.pt `
  --vae-checkpoint ./pretrained_models/sd-vae-ft-ema `
  --modulation --cond_mlp_modulation --rope --finetune_depth 28 `
  --train_stage stage1 --t_min 200 --t_max 700 `
  --lr_dit 1e-5 --lr_new 1e-4 `
  --lambda_latent 0.5 --lambda_app 0.5 --lambda_edge 0.2 --lambda_seg 0.5 --lambda_lpips 0.1
```

stage2 只需把：

```powershell
--train_stage stage2
```

---

## 10) 迁移顺序（避免一次改崩）

1. 先落 `dataset.py + scripts/make_paired_split.py`。
2. 再落 `editing/*.py` 三个新增模块。
3. 再改 `train.py`（import、model、dataset、train loop、args）。
4. 最后替换 `sample_edit.py` 和 `requirements.txt`。

---

## 11) 注意事项（复制时）

- `sample_edit.py` 中的 reverse loop 是最简可跑版本，后续可替换成你现有 `inversion` 框架的等价实现。
- 如果 `SegFormer` 下载慢，可先离线缓存模型后再训练。
- `WeatherEditWrapper` 的 `seg_channels=19` 针对 cityscapes，换 seg 模型时记得同步。

