# BBox 编码实现方案

## 概述

本文档详细说明如何在 DriveDiTFit 中添加 BBox 编码功能，使模型能够根据指定的边界框位置生成车辆。

### 当前状态 vs 目标状态

| 方面 | 当前 | 目标 |
|------|------|------|
| 天气控制 | ✅ 有 | ✅ 保持 |
| 车辆位置控制 | ❌ 无 | ✅ 新增 |
| 输入 | 噪声 + 天气标签 | 噪声 + 天气标签 + BBox |

---

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `drivefit_models.py` | 新增 BBoxEncoder 类，修改 DiT 接收 bbox 输入 |
| `train.py` | 修改训练循环，传入 bbox 数据 |
| `dataset.py` | 修改数据加载，返回 bbox 坐标 |
| `sample.py` | 修改推理代码，支持 bbox 条件 |

---

## 第一部分：drivefit_models.py 修改

### 1.1 新增 BBoxEncoder 类

**位置**：在 `LabelEmbedder` 类之后添加（约 L452）

```python
# ============ 原代码位置 ============
class LabelEmbedder(nn.Module):
    ...
    # LabelEmbedder 类结束

# ============ 在此处新增以下代码 ============

class BBoxEncoder(nn.Module):
    """
    将边界框坐标编码为向量表示，用于条件控制。
    
    输入: bboxes - [B, max_boxes, 4] 归一化坐标 (x1, y1, x2, y2)
    输出: bbox_embed - [B, hidden_size] 聚合后的bbox嵌入
    
    设计思路:
    1. 将每个bbox的4个坐标通过MLP编码为向量
    2. 对多个bbox取平均得到聚合表示
    3. 无bbox时返回可学习的null embedding
    """
    
    def __init__(self, hidden_size, max_boxes=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_boxes = max_boxes
        
        # 坐标编码MLP: [4] -> [hidden_size]
        # 4个输入: x1, y1, x2, y2 (归一化到0-1)
        self.coord_mlp = nn.Sequential(
            nn.Linear(4, hidden_size // 4),   # 4 -> 288
            nn.SiLU(),
            nn.Linear(hidden_size // 4, hidden_size // 2),  # 288 -> 576
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size),  # 576 -> 1152
        )
        
        # 无bbox时的null embedding (类似CFG的null token)
        self.null_embed = nn.Parameter(torch.zeros(1, hidden_size))
        nn.init.normal_(self.null_embed, std=0.02)
        
    def forward(self, bboxes, bbox_mask=None):
        """
        Args:
            bboxes: [B, max_boxes, 4] 边界框坐标，归一化到0-1
            bbox_mask: [B, max_boxes] 有效框掩码，1=有效，0=padding
            
        Returns:
            bbox_embed: [B, hidden_size] 聚合后的bbox嵌入
        """
        B = bboxes.shape[0]
        
        if bbox_mask is None:
            # 如果没有mask，假设所有非零框都有效
            bbox_mask = (bboxes.sum(dim=-1) != 0).float()  # [B, max_boxes]
        
        # 编码每个bbox
        # bboxes: [B, max_boxes, 4] -> [B, max_boxes, hidden_size]
        bbox_embeds = self.coord_mlp(bboxes)
        
        # 加权平均 (根据mask)
        # bbox_mask: [B, max_boxes] -> [B, max_boxes, 1]
        mask_expanded = bbox_mask.unsqueeze(-1)
        
        # 计算每个样本有效框的数量
        num_valid = bbox_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
        
        # 加权求和再平均
        bbox_embed = (bbox_embeds * mask_expanded).sum(dim=1) / num_valid  # [B, hidden_size]
        
        # 对于没有任何有效框的样本，使用null embedding
        no_box_mask = (bbox_mask.sum(dim=1) == 0).unsqueeze(-1)  # [B, 1]
        bbox_embed = torch.where(no_box_mask, self.null_embed.expand(B, -1), bbox_embed)
        
        return bbox_embed
```

---

### 1.2 修改 DiT 类的 __init__

**位置**：`DiT.__init__` 方法中（约 L900-950）

```python
# ============ 原代码 ============
class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        modulation=False,
        patch_modulation=False,
        block_mlp_modulation=False,
        cond_mlp_modulation=False,
        rank=2,
        scenario_num=0,
        rope=False,
        finetune_depth=28,
    ):
        super().__init__()
        # ... 现有初始化代码 ...
        
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob, scenario_num
        )

# ============ 修改后代码 ============
class DiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        modulation=False,
        patch_modulation=False,
        block_mlp_modulation=False,
        cond_mlp_modulation=False,
        rank=2,
        scenario_num=0,
        rope=False,
        finetune_depth=28,
        use_bbox_cond=False,  # ← 新增参数
        max_boxes=10,         # ← 新增参数
    ):
        super().__init__()
        # ... 现有初始化代码 ...
        
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob, scenario_num
        )
        
        # ============ 新增 BBox 编码器 ============
        self.use_bbox_cond = use_bbox_cond
        if use_bbox_cond:
            self.bbox_encoder = BBoxEncoder(hidden_size, max_boxes)
```

---

### 1.3 修改 DiT 类的 forward 方法

**位置**：`DiT.forward` 方法（约 L980-1010）

```python
# ============ 原代码 ============
def forward(self, x, t, y):
    """
    Forward pass of DiT.
    x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
    t: (N,) tensor of diffusion timesteps
    y: (N,) tensor of class labels
    """
    x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
    t = self.t_embedder(t)                   # (N, D)
    y = self.y_embedder(y, self.training)    # (N, D)
    c = t + y                                # (N, D)
    
    for block in self.blocks:
        x = block(x, c)
    x = self.final_layer(x, c)
    x = self.unpatchify(x)
    return x

# ============ 修改后代码 ============
def forward(self, x, t, y, bboxes=None, bbox_mask=None):
    """
    Forward pass of DiT with optional BBox conditioning.
    
    Args:
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels (weather/scenario)
        bboxes: (N, max_boxes, 4) optional bbox coordinates, normalized to [0,1]
        bbox_mask: (N, max_boxes) optional mask for valid bboxes
        
    Returns:
        (N, C, H, W) predicted noise (and variance if learn_sigma)
    """
    x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
    t = self.t_embedder(t)                   # (N, D)
    y = self.y_embedder(y, self.training)    # (N, D)
    
    # ============ 新增：融合 BBox 条件 ============
    if self.use_bbox_cond and bboxes is not None:
        bbox_embed = self.bbox_encoder(bboxes, bbox_mask)  # (N, D)
        c = t + y + bbox_embed  # 融合三种条件
    else:
        c = t + y
    # ============ 新增结束 ============
    
    for block in self.blocks:
        x = block(x, c)
    x = self.final_layer(x, c)
    x = self.unpatchify(x)
    return x
```

---

### 1.4 修改 forward_with_cfg 方法

**位置**：`DiT.forward_with_cfg` 方法（约 L1020-1050）

```python
# ============ 原代码 ============
def forward_with_cfg(self, x, t, y, cfg_scale):
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = self.forward(combined, t, y)
    eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)

# ============ 修改后代码 ============
def forward_with_cfg(self, x, t, y, cfg_scale, bboxes=None, bbox_mask=None):
    """
    Forward with Classifier-Free Guidance, supporting BBox condition.
    
    Args:
        x: (2N, C, H, W) - 前半是条件采样，后半是无条件采样
        t: (2N,) timesteps
        y: (2N,) labels - 后半应该是 null token (scenario_num)
        cfg_scale: guidance scale
        bboxes: (2N, max_boxes, 4) - 后半应该全是0 (无bbox条件)
        bbox_mask: (2N, max_boxes) - 后半应该全是0
    """
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    
    # ============ 修改：传入 bbox 参数 ============
    model_out = self.forward(combined, t, y, bboxes, bbox_mask)
    # ============ 修改结束 ============
    
    eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)
```

---

## 第二部分：dataset.py 修改

### 2.1 修改 CustomImageFolder 返回 bbox 坐标

```python
# ============ 原代码 ============
class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        
        info, target_info, fname = self.info[index]
        box_info = np.load(info)  # 加载掩码 (32, 32)
        
        # ... 数据增强 ...
        
        return img, target, box_info, fname

# ============ 修改后代码 ============
class CustomImageFolder(ImageFolder):
    def __init__(self, root, info_file_path, bbox_coord_path=None, 
                 trans_flip=False, transform=None, target_transform=None, 
                 max_boxes=10):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.info = load_info(info_file_path)
        self.trans_flip = trans_flip
        self.p = 0.5
        
        # ============ 新增：加载 bbox 坐标 ============
        self.max_boxes = max_boxes
        self.bbox_coord_path = bbox_coord_path
        if bbox_coord_path is not None:
            self.bbox_coords = load_info(bbox_coord_path)  # 加载坐标文件
        else:
            self.bbox_coords = None
        # ============ 新增结束 ============
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)
        
        info, target_info, fname = self.info[index]
        box_info = np.load(info)  # 掩码 (32, 32)
        
        # ============ 新增：加载 bbox 坐标 ============
        if self.bbox_coords is not None:
            coord_path, _, _ = self.bbox_coords[index]
            coords = np.load(coord_path)  # (N, 4) 原始坐标
            
            # 归一化到 [0, 1]
            coords = coords.copy()
            coords[:, [0, 2]] /= 32.0  # x 归一化
            coords[:, [1, 3]] /= 32.0  # y 归一化
            
            # Padding 到固定长度
            num_boxes = min(len(coords), self.max_boxes)
            bbox_padded = np.zeros((self.max_boxes, 4), dtype=np.float32)
            bbox_mask = np.zeros(self.max_boxes, dtype=np.float32)
            
            bbox_padded[:num_boxes] = coords[:num_boxes]
            bbox_mask[:num_boxes] = 1.0
        else:
            bbox_padded = np.zeros((self.max_boxes, 4), dtype=np.float32)
            bbox_mask = np.zeros(self.max_boxes, dtype=np.float32)
        # ============ 新增结束 ============
        
        assert target == target_info
        
        if self.trans_flip:
            if torch.rand(1) < self.p:
                img = transforms.functional.hflip(img)
                box_info = np.flip(box_info, axis=1).copy()
                # ============ 新增：同步翻转 bbox 坐标 ============
                if bbox_mask.sum() > 0:
                    # 翻转 x 坐标: x_new = 1 - x_old
                    bbox_padded[:, [0, 2]] = 1.0 - bbox_padded[:, [2, 0]]
                # ============ 新增结束 ============
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # ============ 修改返回值 ============
        return img, target, box_info, fname, bbox_padded, bbox_mask
```

---

## 第三部分：train.py 修改

### 3.1 修改模型创建

```python
# ============ 原代码 ============
model = DiT_models[args.model](
    input_size=latent_size,
    num_classes=args.num_classes,
    modulation=args.modulation,
    # ... 其他参数 ...
)

# ============ 修改后代码 ============
model = DiT_models[args.model](
    input_size=latent_size,
    num_classes=args.num_classes,
    modulation=args.modulation,
    # ... 其他参数 ...
    use_bbox_cond=args.use_bbox_cond,  # ← 新增
    max_boxes=args.max_boxes,           # ← 新增
)
```

### 3.2 修改训练循环

```python
# ============ 原代码 ============
for data in loader:
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
    
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    
    model_kwargs = dict(y=y)
    loss_dict = diffusion.training_losses(model, x, t, boxes_mask, args.mask_rl, model_kwargs)

# ============ 修改后代码 ============
for data in loader:
    if args.boxes_path is None:
        x, y = data
        boxes_mask = None
        bbox_coords = None
        bbox_valid_mask = None
    else:
        # ============ 修改：解包新增的返回值 ============
        x, y, boxes_mask, _, bbox_coords, bbox_valid_mask = data
        boxes_mask = boxes_mask.unsqueeze(1).to(device)
        bbox_coords = bbox_coords.to(device)       # (B, max_boxes, 4)
        bbox_valid_mask = bbox_valid_mask.to(device)  # (B, max_boxes)
        # ============ 修改结束 ============
    
    x = x.to(device)
    y = y.to(device)
    
    with torch.no_grad():
        x = vae.encode(x).latent_dist.sample().mul_(0.18215)
    
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    
    # ============ 修改：传入 bbox 参数 ============
    model_kwargs = dict(y=y, bboxes=bbox_coords, bbox_mask=bbox_valid_mask)
    # ============ 修改结束 ============
    
    loss_dict = diffusion.training_losses(model, x, t, boxes_mask, args.mask_rl, model_kwargs)
```

### 3.3 新增命令行参数

```python
# ============ 在 argparse 部分新增 ============
parser.add_argument("--use_bbox_cond", action="store_true",
                    help="Enable BBox conditioning for layout control")
parser.add_argument("--max_boxes", type=int, default=10,
                    help="Maximum number of bboxes per image")
parser.add_argument("--bbox_coord_path", type=str, default=None,
                    help="Path to bbox coordinate files (optional)")
```

---

## 第四部分：推理使用示例

### 4.1 指定位置生成车辆

```python
import torch
from drivefit_models import DiT_models

# 加载模型
model = DiT_models["DiT-XL/2"](
    use_bbox_cond=True,
    max_boxes=10,
    scenario_num=5,
    # ... 其他参数
)
model.load_state_dict(...)
model.eval()

# 准备输入
z = torch.randn(1, 4, 32, 32)  # 随机噪声
y = torch.tensor([0])  # 天气: sunny

# 定义车辆位置 (归一化坐标)
# 格式: [x1, y1, x2, y2]，范围 [0, 1]
bboxes = torch.zeros(1, 10, 4)
bboxes[0, 0] = torch.tensor([0.1, 0.4, 0.3, 0.6])  # 左侧一辆车
bboxes[0, 1] = torch.tensor([0.7, 0.4, 0.9, 0.6])  # 右侧一辆车

bbox_mask = torch.zeros(1, 10)
bbox_mask[0, 0] = 1  # 第1个框有效
bbox_mask[0, 1] = 1  # 第2个框有效

# CFG 准备 (条件 + 无条件)
z = torch.cat([z, z], 0)
y = torch.cat([y, torch.tensor([5])], 0)  # 5 = null token
bboxes = torch.cat([bboxes, torch.zeros_like(bboxes)], 0)
bbox_mask = torch.cat([bbox_mask, torch.zeros_like(bbox_mask)], 0)

# 采样
with torch.no_grad():
    samples = diffusion.p_sample_loop(
        lambda *args, **kwargs: model.forward_with_cfg(
            *args, bboxes=bboxes, bbox_mask=bbox_mask, **kwargs
        ),
        z.shape, z,
        model_kwargs=dict(y=y, cfg_scale=4.0)
    )
```

---

## 第五部分：需要额外生成的文件

### 5.1 新增 bbox 坐标提取脚本

需要修改 `scripts/extract_boxes.py`，除了生成掩码外，还需要生成坐标文件：

```python
# 在 extract_boxes.py 中新增

# 保存坐标 (除了掩码)
coord_save_dir = os.path.join(OUTPUT_PATH + "_coords", weather)
os.makedirs(coord_save_dir, exist_ok=True)

# 保存格式: (N, 4) 的数组，N是该帧的车辆数
if name in object_dict:
    coords = np.array(object_dict[name], dtype=np.float32)
else:
    coords = np.zeros((0, 4), dtype=np.float32)

np.save(os.path.join(coord_save_dir, f"{name}.npy"), coords)
```

---

## 训练命令示例

```bash
python train.py --model DiT-XL/2 \
    --data-path ./datasets/Ithaca365/Ithaca365-scenario \
    --boxes-path ./datasets/box_info \
    --bbox_coord_path ./datasets/box_info_coords \
    --use_bbox_cond \
    --max_boxes 10 \
    --epochs 3000 \
    --global-batch-size 4 \
    --modulation --cond_mlp_modulation --rope
```

---

## 注意事项

1. **BBox 坐标归一化**：所有坐标归一化到 [0, 1]，相对于 latent space (32×32)
2. **max_boxes 选择**：根据数据集统计，选择合适的最大框数
3. **CFG 采样**：无条件部分的 bbox 应该全为 0
4. **兼容性**：当 `use_bbox_cond=False` 或 `bboxes=None` 时，回退到原有行为
5. **翻转增强**：水平翻转时需要同步翻转 bbox 坐标
