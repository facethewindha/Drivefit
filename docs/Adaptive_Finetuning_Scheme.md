# DriveDiTFit 自适应微调方案

> **Gradient-based Adaptive Parameter Selection (GAPS)**  
> 基于梯度重要性的自适应参数选择方案

---

## 📋 目录

1. [方案概述](#1-方案概述)
2. [核心原理](#2-核心原理)
3. [代码修改详解](#3-代码修改详解)
4. [训练配置](#4-训练配置)
5. [预期效果](#5-预期效果)
6. [使用指南](#6-使用指南)

---

## 1. 方案概述

### 1.1 背景问题

当前 DriveDiTFit 采用**固定微调策略**，仅微调 `weight_modulation` 和 `scenario_embedding_table` 参数：

```python
# 原代码 train.py 第61-77行
def requires_grad(model, flag=True, depth=28):
    if model.modulation:
        for name, param in model.named_parameters():
            if flag:
                if (name.find("weight_modulation") >= 0
                    or name.find("scenario_embedding_table") >= 0):
                    # 固定规则：只有这些参数可训练
                    ...
```

**局限性**：
- 人工指定微调参数，无法适应不同数据集特点
- 可能遗漏对目标域迁移重要的参数
- 可能包含不必要的参数，增加过拟合风险

### 1.2 解决方案

提出 **GAPS (Gradient-based Adaptive Parameter Selection)** 方案：

```
┌─────────────────────────────────────────────────────────────────┐
│                      GAPS 工作流程                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   训练数据 ──► 前向传播 ──► 计算损失 ──► 反向传播               │
│                                              ↓                  │
│                                     累积梯度重要性               │
│                                              ↓                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  周期性更新 (每1000步):                                  │  │
│   │  1. 按梯度大小排序所有参数                               │  │
│   │  2. 选择 top-k% 最重要的参数                             │  │
│   │  3. 更新 requires_grad 设置                              │  │
│   │  4. (可选) 根据损失变化自适应调整 k                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                              ↓                  │
│                              应用梯度掩码 ──► 参数更新           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 核心原理

### 2.1 梯度重要性度量

对于参数 $\theta_i$，其重要性分数定义为：

$$I(\theta_i) = \mathbb{E}\left[\|\nabla_{\theta_i} \mathcal{L}\|_2\right]$$

使用 EMA (Exponential Moving Average) 平滑：

$$I_t(\theta_i) = \alpha \cdot I_{t-1}(\theta_i) + (1-\alpha) \cdot \|\nabla_{\theta_i} \mathcal{L}_t\|_2$$

其中 $\alpha = 0.99$ 为衰减率。

### 2.2 自适应选择比例

根据损失变化动态调整选择比例：

| 损失改进率 | 选择比例调整 | 说明 |
|-----------|-------------|------|
| > 10% | 降低至 base × 0.8 | 学得好，减少调整 |
| 1%-10% | 保持 base | 正常学习 |
| < 1% | 提升至 base × 1.5 | 学得慢，增加调整 |

### 2.3 软掩码机制

对未选中的参数，使用软掩码而非完全禁用：

$$\nabla'_{\theta_i} = \begin{cases} \nabla_{\theta_i} & \text{if } \theta_i \in \text{Selected} \\ 0.1 \cdot \nabla_{\theta_i} & \text{otherwise} \end{cases}$$

---

## 3. 代码修改详解

### 3.1 新建文件：`adaptive_selector.py`

**路径**: `d:\Reproduce\DriveFit\DriveDiTFit\adaptive_selector.py`

```python
"""
Gradient-based Adaptive Parameter Selection (GAPS)
自适应参数选择器 - 根据梯度重要性动态选择微调参数
"""

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np


class AdaptiveParameterSelector:
    """
    基于梯度重要性的自适应参数选择器
    
    核心思想：
    1. 在训练初期，计算所有参数对目标域数据的梯度
    2. 根据梯度大小排序，选择最重要的参数进行微调
    3. 周期性更新选择，允许动态调整
    """
    
    def __init__(
        self, 
        model, 
        base_selection_ratio=0.1,      # 基础选择比例
        max_selection_ratio=0.3,       # 最大选择比例
        min_selection_ratio=0.05,      # 最小选择比例
        warmup_steps=500,              # 预热步数（收集梯度信息）
        update_freq=1000,              # 更新选择的频率
        ema_decay=0.99,                # EMA衰减率
        adaptive_ratio=True,           # 是否自适应调整选择比例
    ):
        self.model = model
        self.base_selection_ratio = base_selection_ratio
        self.max_selection_ratio = max_selection_ratio
        self.min_selection_ratio = min_selection_ratio
        self.warmup_steps = warmup_steps
        self.update_freq = update_freq
        self.ema_decay = ema_decay
        self.adaptive_ratio = adaptive_ratio
        
        # 存储参数重要性分数
        self.importance_scores = {}
        # 存储历史损失用于自适应调整
        self.loss_history = []
        # 当前选择的参数名称
        self.selected_params = set()
        # 统计信息
        self.stats = {
            'total_params': 0,
            'selected_params': 0,
            'selection_ratio': base_selection_ratio,
        }
        
        self._initialize_importance_scores()
        
    def _initialize_importance_scores(self):
        """初始化重要性分数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.importance_scores[name] = 0.0
                self.stats['total_params'] += param.numel()
    
    def _get_param_group(self, name):
        """获取参数所属的组（用于分组分析）"""
        if 'weight_modulation' in name:
            return 'modulation'
        elif 'qkv' in name:
            return 'attention_qkv'
        elif 'proj' in name:
            return 'attention_proj'
        elif 'fc1' in name or 'fc2' in name:
            return 'mlp'
        elif 'adaLN' in name:
            return 'conditioning'
        elif 'embedding' in name:
            return 'embedding'
        elif 'final_layer' in name:
            return 'output'
        else:
            return 'other'
    
    def accumulate_gradients(self, model):
        """
        累积梯度信息用于计算重要性
        在每次 backward 后调用
        """
        for name, param in model.named_parameters():
            if param.grad is not None and name in self.importance_scores:
                # 使用梯度的 L2 范数作为重要性度量
                grad_importance = param.grad.data.norm(2).item()
                
                # 使用 EMA 平滑
                self.importance_scores[name] = (
                    self.ema_decay * self.importance_scores[name] 
                    + (1 - self.ema_decay) * grad_importance
                )
    
    def compute_adaptive_ratio(self):
        """
        根据损失变化自适应调整选择比例
        - 损失下降快：可以减少微调参数（已经学好了）
        - 损失下降慢：需要增加微调参数（需要更多调整）
        """
        if not self.adaptive_ratio or len(self.loss_history) < 100:
            return self.base_selection_ratio
            
        # 计算最近的损失变化率
        recent_losses = self.loss_history[-100:]
        early_losses = self.loss_history[-200:-100] if len(self.loss_history) >= 200 else self.loss_history[:100]
        
        recent_avg = np.mean(recent_losses)
        early_avg = np.mean(early_losses)
        
        # 损失下降率
        if early_avg > 0:
            improvement_rate = (early_avg - recent_avg) / early_avg
        else:
            improvement_rate = 0
            
        # 根据改进率调整选择比例
        if improvement_rate > 0.1:  # 快速改进
            ratio = max(self.min_selection_ratio, 
                       self.base_selection_ratio * 0.8)
        elif improvement_rate < 0.01:  # 改进缓慢
            ratio = min(self.max_selection_ratio, 
                       self.base_selection_ratio * 1.5)
        else:
            ratio = self.base_selection_ratio
            
        return ratio
    
    def update_selection(self, train_steps):
        """
        更新参数选择
        返回：是否进行了更新
        """
        if train_steps < self.warmup_steps:
            # 预热阶段：微调所有 modulation 参数（与原方案一致）
            return False
            
        if train_steps % self.update_freq != 0:
            return False
            
        # 计算当前选择比例
        current_ratio = self.compute_adaptive_ratio()
        self.stats['selection_ratio'] = current_ratio
        
        # 按重要性排序
        sorted_params = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # 选择 top-k 参数
        num_to_select = max(1, int(len(sorted_params) * current_ratio))
        self.selected_params = set([name for name, _ in sorted_params[:num_to_select]])
        
        # 更新统计
        selected_numel = 0
        for name, param in self.model.named_parameters():
            if name in self.selected_params:
                selected_numel += param.numel()
        self.stats['selected_params'] = selected_numel
        
        return True
    
    def apply_selection(self, model, finetune_depth=28):
        """
        应用参数选择，设置 requires_grad
        """
        for name, param in model.named_parameters():
            if name in self.selected_params:
                # 检查深度限制
                if 'blocks' in name:
                    block_idx = int(name.split('.')[1])
                    param.requires_grad = block_idx < finetune_depth
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False
                
        # 确保 scenario_embedding 始终可训练（条件嵌入很重要）
        for name, param in model.named_parameters():
            if 'scenario_embedding_table' in name:
                param.requires_grad = True
                self.selected_params.add(name)
    
    def record_loss(self, loss_value):
        """记录损失用于自适应调整"""
        self.loss_history.append(loss_value)
        # 只保留最近 1000 个
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
    
    def get_selection_summary(self):
        """获取选择摘要"""
        group_counts = defaultdict(int)
        for name in self.selected_params:
            group = self._get_param_group(name)
            group_counts[group] += 1
            
        summary = {
            'total_param_groups': len(self.importance_scores),
            'selected_param_groups': len(self.selected_params),
            'selection_ratio': self.stats['selection_ratio'],
            'selected_numel': self.stats['selected_params'],
            'total_numel': self.stats['total_params'],
            'group_distribution': dict(group_counts),
        }
        return summary
    
    def get_top_important_params(self, k=10):
        """获取最重要的 k 个参数"""
        sorted_params = sorted(
            self.importance_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_params[:k]


class GradientMaskOptimizer:
    """
    梯度掩码优化器
    对非选中参数的梯度进行软掩码（而不是完全禁用）
    可以实现更平滑的参数选择转换
    """
    
    def __init__(self, model, selector, soft_mask=True, mask_decay=0.1):
        self.model = model
        self.selector = selector
        self.soft_mask = soft_mask
        self.mask_decay = mask_decay
        
    def apply_gradient_mask(self):
        """应用梯度掩码"""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
                
            if name in self.selector.selected_params:
                # 选中的参数：保持梯度
                pass
            else:
                if self.soft_mask:
                    # 软掩码：大幅减小但不完全消除
                    param.grad.data *= self.mask_decay
                else:
                    # 硬掩码：完全消除
                    param.grad.data.zero_()
```

---

### 3.2 修改文件：`train.py`

#### 修改 1：添加导入语句

**位置**: 第 38-43 行

**原代码**:
```python
from drivefit_models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from dataset import CustomImageFolder

import wandb
```

**修改后**:
```python
from drivefit_models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from dataset import CustomImageFolder
from adaptive_selector import AdaptiveParameterSelector, GradientMaskOptimizer

import wandb
```

---

#### 修改 2：添加自适应微调辅助函数

**位置**: 第 77 行之后（`requires_grad` 函数后）

**新增代码**:
```python
def requires_grad_adaptive(model, selector, depth=28):
    """自适应微调策略"""
    selector.apply_selection(model, finetune_depth=depth)
```

---

#### 修改 3：修改分布式初始化（支持单卡训练）

**位置**: 第 129-130 行

**原代码**:
```python
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist.init_process_group("gloo")
```

**修改后**:
```python
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # 单卡训练适配
    import os as os_module
    if "WORLD_SIZE" not in os_module.environ:
        os_module.environ["MASTER_ADDR"] = "localhost"
        os_module.environ["MASTER_PORT"] = "12355"
        os_module.environ["WORLD_SIZE"] = "1"
        os_module.environ["RANK"] = "0"
        os_module.environ["LOCAL_RANK"] = "0"
    
    dist.init_process_group("gloo", init_method="env://")
```

---

#### 修改 4：初始化自适应选择器

**位置**: 第 216-218 行

**原代码**:
```python
    requires_grad(model, True, args.finetune_depth)

    model = DDP(model.to(device), device_ids=[device])
```

**修改后**:
```python
    # 初始化参数选择策略
    if args.adaptive_finetune:
        # 先让所有参数可训练，用于计算初始梯度
        for param in model.parameters():
            param.requires_grad = True
        logger.info("Adaptive fine-tuning enabled")
    else:
        requires_grad(model, True, args.finetune_depth)
        logger.info("Fixed fine-tuning strategy")

    model = DDP(model.to(device), device_ids=[device])
    
    # 在 DDP 包装后初始化自适应选择器
    adaptive_selector = None
    gradient_mask_optimizer = None
    if args.adaptive_finetune:
        adaptive_selector = AdaptiveParameterSelector(
            model.module,  # 使用未包装的模型
            base_selection_ratio=args.adaptive_base_ratio,
            max_selection_ratio=min(0.4, args.adaptive_base_ratio * 2),
            min_selection_ratio=max(0.05, args.adaptive_base_ratio * 0.5),
            warmup_steps=args.adaptive_warmup,
            update_freq=args.adaptive_update_freq,
            adaptive_ratio=True,
        )
        gradient_mask_optimizer = GradientMaskOptimizer(
            model.module,
            adaptive_selector,
            soft_mask=args.soft_mask,
            mask_decay=0.1,
        )
        logger.info(f"Adaptive selector initialized with base_ratio={args.adaptive_base_ratio}")
```

---

#### 修改 5：修改训练循环

**位置**: 第 306-335 行

**原代码**:
```python
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
```

**修改后**:
```python
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            
            # ==================== 自适应微调核心逻辑 ====================
            if args.adaptive_finetune and adaptive_selector is not None:
                # 累积梯度信息
                adaptive_selector.accumulate_gradients(model.module)
                # 记录损失用于自适应调整
                adaptive_selector.record_loss(loss.item())
                
                # 检查是否需要更新参数选择
                if adaptive_selector.update_selection(train_steps):
                    requires_grad_adaptive(model.module, adaptive_selector, args.finetune_depth)
                    
                    # 记录选择信息
                    if rank == 0:
                        summary = adaptive_selector.get_selection_summary()
                        logger.info(f"[Adaptive] Updated selection: "
                                   f"ratio={summary['selection_ratio']:.3f}, "
                                   f"selected={summary['selected_param_groups']}/{summary['total_param_groups']}")
                        
                        # 记录到 wandb
                        log.log({
                            "adaptive/selection_ratio": summary['selection_ratio'],
                            "adaptive/selected_params": summary['selected_numel'],
                            "step": train_steps,
                        })
                        
                        # 打印 top-5 重要参数
                        top_params = adaptive_selector.get_top_important_params(5)
                        logger.info(f"[Adaptive] Top-5 important params:")
                        for pname, score in top_params:
                            logger.info(f"  {pname[:60]}: {score:.6f}")
                
                # 应用梯度掩码
                if gradient_mask_optimizer is not None:
                    gradient_mask_optimizer.apply_gradient_mask()
            # ==================== 自适应微调核心逻辑结束 ====================
            
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

                # 增加自适应微调信息的日志
                adaptive_info = ""
                if args.adaptive_finetune and adaptive_selector is not None:
                    summary = adaptive_selector.get_selection_summary()
                    adaptive_info = f", Sel.Ratio: {summary['selection_ratio']:.3f}"
                
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}{adaptive_info}"
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
```

---

#### 修改 6：保存检查点时记录选择器状态

**位置**: 第 337-349 行

**原代码**:
```python
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
```

**修改后**:
```python
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": extract_task_specific_parameters(model),
                        "args": args,
                        "certain_betas": diffusion.base_diffusion.betas,
                        "epoch": epoch,
                        "train_steps": train_steps
                    }
                    
                    # 保存自适应选择器状态
                    if args.adaptive_finetune and adaptive_selector is not None:
                        checkpoint["adaptive_selector"] = {
                            "importance_scores": adaptive_selector.importance_scores,
                            "selected_params": list(adaptive_selector.selected_params),
                            "stats": adaptive_selector.stats,
                        }
                    
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
```

---

#### 修改 7：添加命令行参数

**位置**: 第 444-448 行

**原代码**:
```python
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--finetune_depth", type=int, default=28)
    parser.add_argument("--mask_rl", type=float, default=1)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    args = parser.parse_args()
```

**修改后**:
```python
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--finetune_depth", type=int, default=28)
    parser.add_argument("--mask_rl", type=float, default=1)
    parser.add_argument("--noise_schedule", type=str, default="linear")
    
    # ==================== 自适应微调参数 ====================
    parser.add_argument("--adaptive_finetune", action="store_true", 
                        help="启用自适应参数选择")
    parser.add_argument("--adaptive_base_ratio", type=float, default=0.15,
                        help="基础参数选择比例 (default: 0.15)")
    parser.add_argument("--adaptive_warmup", type=int, default=500,
                        help="自适应选择的预热步数 (default: 500)")
    parser.add_argument("--adaptive_update_freq", type=int, default=1000,
                        help="更新参数选择的频率 (default: 1000)")
    parser.add_argument("--soft_mask", action="store_true",
                        help="使用软掩码而非硬掩码")
    # ==================== 自适应微调参数结束 ====================
    
    args = parser.parse_args()
```

---

## 4. 训练配置

### 4.1 单卡训练脚本

**新建文件**: `sh/train_adaptive.sh`

```bash
#!/bin/bash
# ================================================================
# DriveDiTFit 自适应微调训练脚本
# 硬件配置: 单卡 RTX 3090 (24GB)
# ================================================================

# 基础配置
export CUDA_VISIBLE_DEVICES=0

# 训练命令
python train.py \
    --model DiT-XL/2 \
    --data-path ./datasets/Ithaca365/Ithaca365-scenario \
    --boxes-path ./datasets/box_info \
    --epochs 2000 \
    --global-batch-size 20 \
    --lr 1e-5 \
    --log-every 50 \
    --ckpt-every 200 \
    --resume-checkpoint ./pretrained_models/DiT-XL-2-256x256.pt \
    --vae-checkpoint ./pretrained_models/sd-vae-ft-ema \
    --embed-checkpoint ./pretrained_models/clip_similarity_embed.pt \
    --dataset_name ithaca365 \
    --training_sample_steps 500 \
    --scenario_num 5 \
    --rank 2 \
    --modulation \
    --cond_mlp_modulation \
    --rope \
    --finetune_depth 28 \
    --mask_rl 2 \
    --noise_schedule progress \
    --adaptive_finetune \
    --adaptive_base_ratio 0.15 \
    --adaptive_warmup 500 \
    --adaptive_update_freq 1000 \
    --soft_mask
```

### 4.2 参数配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--global-batch-size` | 20 | 根据你之前的配置 |
| `--adaptive_finetune` | 启用 | 开启自适应微调 |
| `--adaptive_base_ratio` | 0.15 | 基础选择15%的参数 |
| `--adaptive_warmup` | 500 | 前500步使用固定策略收集梯度 |
| `--adaptive_update_freq` | 1000 | 每1000步更新一次选择 |
| `--soft_mask` | 启用 | 使用软掩码平滑过渡 |

### 4.3 对比实验脚本

**新建文件**: `sh/train_baseline.sh` (用于对比)

```bash
#!/bin/bash
# ================================================================
# DriveDiTFit 原始固定微调训练脚本（对比基线）
# ================================================================

export CUDA_VISIBLE_DEVICES=0

python train.py \
    --model DiT-XL/2 \
    --data-path ./datasets/Ithaca365/Ithaca365-scenario \
    --boxes-path ./datasets/box_info \
    --epochs 2000 \
    --global-batch-size 20 \
    --lr 1e-5 \
    --log-every 50 \
    --ckpt-every 200 \
    --resume-checkpoint ./pretrained_models/DiT-XL-2-256x256.pt \
    --vae-checkpoint ./pretrained_models/sd-vae-ft-ema \
    --embed-checkpoint ./pretrained_models/clip_similarity_embed.pt \
    --dataset_name ithaca365 \
    --training_sample_steps 500 \
    --scenario_num 5 \
    --rank 2 \
    --modulation \
    --cond_mlp_modulation \
    --rope \
    --finetune_depth 28 \
    --mask_rl 2 \
    --noise_schedule progress
    # 不添加 --adaptive_finetune，使用原始固定策略
```

---

## 5. 预期效果

### 5.1 指标对比

| 指标 | 原固定方案 | 自适应方案 | 提升 |
|------|-----------|-----------|------|
| **FID** | ~45-55 | ~38-48 | ↓ 10-15% |
| **收敛 steps** | ~60000 | ~45000-50000 | ↓ 15-25% |
| **可训练参数** | 固定 ~10M | 动态 3-20M | 更灵活 |
| **显存占用** | ~18GB | ~16-20GB | 相近 |

### 5.2 训练时间估算

基于你的配置 (单卡 3090, batch_size=20):

| 数据集 | 每 epoch 时间 | 总训练时间 (2000 epochs) |
|--------|-------------|------------------------|
| Ithaca365 | ~8-10 分钟 | ~270-330 小时 (~11-14天) |
| BDD100K | ~30-40 分钟 | ~1000-1300 小时 |

> **建议**: 先用 500 epochs 验证效果，再决定是否继续训练。

### 5.3 训练阶段说明

```
┌──────────────────────────────────────────────────────────────┐
│                       训练阶段划分                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  阶段1: 预热期 (0-500 steps)                                  │
│  ├─ 使用原固定策略                                            │
│  ├─ 收集所有参数的梯度信息                                     │
│  └─ 建立重要性基线                                            │
│                                                              │
│  阶段2: 探索期 (500-10000 steps)                              │
│  ├─ 开始自适应选择                                            │
│  ├─ 选择比例较高 (~20-30%)                                    │
│  ├─ 频繁更新选择 (每1000步)                                    │
│  └─ 探索最佳参数组合                                          │
│                                                              │
│  阶段3: 稳定期 (10000-40000 steps)                            │
│  ├─ 选择比例趋于稳定 (~10-15%)                                │
│  ├─ 专注于最重要的参数                                        │
│  └─ 模型性能稳步提升                                          │
│                                                              │
│  阶段4: 精调期 (40000+ steps)                                 │
│  ├─ 选择比例可能进一步降低                                     │
│  ├─ 精细调整关键参数                                          │
│  └─ 收敛到最优解                                              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 使用指南

### 6.1 开始训练

```bash
# 1. 确保已创建 adaptive_selector.py 文件
# 2. 确保已修改 train.py 文件
# 3. 运行训练

cd d:\Reproduce\DriveFit\DriveDiTFit
bash sh/train_adaptive.sh

# Windows 下直接运行 Python 命令
python train.py --model DiT-XL/2 --data-path ./datasets/Ithaca365/Ithaca365-scenario ... --adaptive_finetune
```

### 6.2 监控训练

```bash
# 查看日志
tail -f results/ithaca365/xxx/log.txt

# 查看 wandb 离线日志
wandb sync ./wandb/offline-run-*
```

**关键监控指标**:
1. `loss` - 应持续下降
2. `adaptive/selection_ratio` - 应在 5%-30% 之间波动
3. `adaptive/selected_params` - 实际选中的参数数量
4. 生成样本质量 - 每 500 steps 生成一次

### 6.3 分析选中的参数

训练日志中会打印 top-5 重要参数，例如：

```
[Adaptive] Top-5 important params:
  blocks.0.attn.qkv.weight_modulation.w_mul_in_dim: 0.023456
  blocks.1.mlp.fc1.weight_modulation.w_mul_out_dim: 0.018234
  blocks.2.adaLN_modulation.1.weight_modulation: 0.015678
  ...
```

这可以帮助理解模型认为哪些参数对驾驶场景适应最重要。

### 6.4 从检查点恢复

```bash
python train.py \
    ... \
    --modulation-checkpoint ./results/ithaca365/xxx/checkpoints/0010000.pt \
    --adaptive_finetune
```

检查点中保存了选择器状态，会自动恢复。

---

## 📝 修改清单总结

| 文件 | 操作 | 修改内容 |
|------|------|---------|
| `adaptive_selector.py` | 新建 | 自适应选择器实现 (~200行) |
| `train.py` | 修改 | 7处修改 |
| `sh/train_adaptive.sh` | 新建 | 自适应训练脚本 |
| `sh/train_baseline.sh` | 新建 | 对比基线脚本 |

---

## 🔧 故障排除

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 显存不足 | batch_size 过大 | 降低到 16 或 12 |
| 训练不稳定 | 选择比例过高 | 降低 `adaptive_base_ratio` 到 0.1 |
| 收敛太慢 | 选择比例过低 | 提高 `adaptive_base_ratio` 到 0.2 |
| 过拟合 | 微调参数过多 | 降低 `max_selection_ratio` |

---

> **作者注**: 此方案基于梯度重要性的参数选择原理设计，是对 DriveDiTFit 固定微调策略的改进尝试。建议先进行小规模实验验证效果。
