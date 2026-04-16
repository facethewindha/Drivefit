# DriveDiTFit Weather Editing 重构方案（评审版）

> 状态：仅方案文档，不改代码。
> 目标：将现有“8通道拼接 + identity混合”训练，重构为“paired supervised 的 sunny->rain 编辑”主线。

## 1. 最终目标与边界

### 1.1 任务定义
输入：晴天图 `x_s`  
输出：同场景雨天图 `x_hat`  
监督：paired 雨天真值 `x_r`

### 1.2 本版范围（第一版）
- 只做单任务：`sunny -> rain`
- 只用 paired 数据（1225 对）
- 不使用额外 350 张 unpaired sunny
- 保留扩散噪声预测框架（DiT 预测 `eps`）
- 移除“clean source latent 直接通道拼接”
- 增加 Source/Structure 条件分支 + 结构一致性损失

### 1.3 冻结与训练
- 冻结：VAE、Segmentation 模型
- 训练：DiT（微调）、SrcEncoder、StrEncoder、条件注入层

---

## 2. 数据方案（明确改动）

## 2.1 划分策略（固定）
从 1225 对 paired 中按固定随机种子划分：
- `train: 980`
- `val: 122`
- `test: 123`

## 2.2 新增文件
- `datasets/splits/sunny_rain_train.txt`
- `datasets/splits/sunny_rain_val.txt`
- `datasets/splits/sunny_rain_test.txt`

每行一个样本 ID（可映射为 sunny/rain 同名文件）。

## 2.3 代码改动点

### 文件：`dataset.py`

#### 现状
- `WeatherEditDataset` 支持 `identity_ratio` 混合采样。
- 编辑样本是 unpaired 随机 target。

#### 改成
新增 `PairedWeatherDataset`（保留旧类，不删除）：
- 输入 split 文件（train/val/test）
- 直接返回 paired：`src_img=x_s`, `tgt_img=x_r`
- 返回字段：
  - `src_img`
  - `tgt_img`
  - `src_label`（固定 sunny=0）
  - `tgt_label`（固定 rain=1）
  - `sample_id`

#### 不再使用
- 第一版训练中不再依赖 `identity_ratio`
- 不再返回 `is_identity`

---

## 3. 模型结构方案（明确改动）

## 3.1 总体结构变更

### 现状
- 训练输入：`cat([z_tgt_noisy, z_src], dim=1)`（8通道）
- 模型 `PatchEmbed` 按 `use_src_cond` 切换 4/8 通道。

### 改成
- 训练输入：仅 `z_s_noisy`（4通道）
- `z_src` 不再直接拼接输入
- 条件信息通过两路 encoder + cross-attention 注入

## 3.2 新增模块

### 文件：`drivefit_models.py`

#### 新增 A：`SourceEncoder`
- 输入：`x_s`（RGB）
- 输出：`src_tokens`（用于 cross-attn context）
- 推荐轻量结构：Conv/Downsample + 1x1 proj + flatten token

#### 新增 B：`StructureEncoder`
- 输入：`[seg_logits(x_s), edge_map(x_s)]`
- 输出：`str_tokens`

#### 新增 C：`ConditionFusion`
- 将 `src_tokens` 与 `str_tokens` 融合成 `ctx_tokens`
- 方案：concat 后线性投影（第一版足够）

#### 新增 D：DiTBlock 条件注入
- 在若干 DiT block（建议后 1/3 block）加入 cross-attn：
  - Query：当前 DiT token
  - Key/Value：`ctx_tokens`
- 保留原 adaLN/weather embedding 机制

#### 接口签名改动
`DiT.forward(...)` 从：
- `forward(self, x, t, y=None, y_src=None, y_tgt=None)`

改为：
- `forward(self, x, t, y_tgt=None, src_tokens=None, str_tokens=None)`

说明：
- 第一版单任务可固定 `y_tgt=rain`，但接口保留参数便于未来多天气扩展。

#### 删除/停用逻辑
- 停用 `use_src_cond` 对 8通道的分支路径。
- 停用 `x_embedder.proj.weight` 的 4->8 拼接加载补丁逻辑（训练主线不再需要）。

---

## 4. 结构先验提取方案（明确改动）

## 4.1 Seg 方案（已定）
- 使用：`nvidia/segformer-b0-finetuned-cityscapes-1024-1024`
- 运行方式：冻结推理，仅用于先验与 `L_seg`

## 4.2 Edge 方案（第一版）
- 使用 Sobel（可微且依赖少）
- 由 `x_s` 与 `x_hat` 分别提取边缘图用于 `L_edge`

## 4.3 新增文件建议
- `models/seg_extractor.py`：封装 SegFormer 推理（冻结）
- `utils/edge.py`：Sobel 边缘提取

---

## 5. 训练流程改造（明确改动）

### 文件：`train.py`

#### 现状主干（image_editing 分支）
- `z_src = Enc(src_img)`
- `z_tgt = Enc(tgt_img)`
- 对 `z_tgt` 加噪 -> `z_tgt_noisy`
- 输入 `x_input = cat([z_tgt_noisy, z_src])`
- loss 以 `L_diff + lambda_id * L_id + L_var` 为主

#### 改成主流程
给 paired 样本 `(x_s, x_r)`：
1. `z_s = Enc(x_s)`, `z_r = Enc(x_r)`
2. `t ~ Uniform([200,700])`
3. `z_s_noisy = q_sample(z_s, t, eps)`
4. 条件提取：
   - `src_tokens = SourceEncoder(x_s)`
   - `seg_logits = SegExtractor(x_s)`
   - `edge_src = Sobel(x_s)`
   - `str_tokens = StructureEncoder(concat(seg_logits, edge_src))`
5. 预测噪声：
   - `eps_hat = model(z_s_noisy, t, y_tgt=rain, src_tokens, str_tokens)`
6. 反推 `z0_hat`：
   - `z0_hat = predict_xstart_from_eps(z_s_noisy, t, eps_hat)`
7. 解码：`x_hat = Dec(z0_hat)`
8. 计算总损失并反传

#### 新损失（第一版）
- `L_diff = ||eps_hat - eps||_2^2`
- `L_latent = ||z0_hat - z_r||_1`
- `L_app = ||x_hat - x_r||_1 + 0.1*LPIPS(x_hat, x_r)`
- `L_edge = ||Sobel(x_hat) - Sobel(x_s)||_1`
- `L_seg = ||Seg(x_hat) - Seg(x_s)||_1`

总损失：
`L = L_diff + 0.5*L_latent + 0.5*L_app + 0.2*L_edge + 0.5*L_seg`

#### 两阶段训练（已确认）
- **阶段 A（稳定对齐）**：`L = L_diff + 0.5*L_latent`
- **阶段 B（外观+结构）**：启用完整损失

#### 优化器分组
- DiT 参数学习率：`1e-5 ~ 5e-5`
- 新增条件模块学习率：`1e-4`

#### 参数新增建议
- `--task_type image_editing_paired`
- `--split_dir datasets/splits`
- `--train_stage {stage1,stage2}`
- `--t_min 200 --t_max 700`
- `--lambda_latent --lambda_app --lambda_edge --lambda_seg --lambda_lpips`

#### 参数停用建议
- `--identity_ratio`（第一版不使用）
- `--use_src_cond`（8通道拼接路径停用）

---

## 6. 推理流程改造（明确改动）

### 文件：`sample_edit.py`

#### 现状
- 参数与 `BaseInverter.edit` 调用不一致（旧接口痕迹）
- 条件仍有 `y` 单标签遗留

#### 改成
输入一张 `x_s`：
1. `z_s = Enc(x_s)`
2. `src_tokens/str_tokens` 与训练同路径提取
3. 加噪到 `t_edit`（建议 300~450 可调）
4. reverse denoise 时始终传：
   - `y_tgt=rain`
   - `src_tokens`
   - `str_tokens`
5. 输出 `x_hat = Dec(z0_hat)`

#### CLI 建议
- `--target_weather rain`（第一版默认固定）
- `--t_edit 350`
- `--cfg_scale`（若保留 CFG）
- `--inversion_steps --denoise_steps`

---

## 7. Inversion/采样接口一致性（明确改动）

### 文件：`inversion/base_inverter.py`
- 统一 `edit(...)` 参数：
  - `source_weather_label`
  - `target_weather_label`
  - `source_conditions`（src/str tokens 或其上游输入）
- 不再依赖旧 `y` 字段。

### 文件：`inversion/ddim_inverter.py`
- `model_fn` 调用统一新签名：
  - `model(x, t, y_tgt=..., src_tokens=..., str_tokens=...)`
- 移除 8通道拼接路径。

---

## 8. 依赖与工程文件改动

### 文件：`requirements.txt`
新增/确认：
- `transformers`
- `lpips`
- `opencv-python`（如 edge/可视化需要）

### 文件：`sh/train.sh`
改为 paired 编辑训练命令，显式传 split 与 stage 参数。

---

## 9. 验证与测试方案

## 9.1 验证集（122 对）
每 `N` step 评估：
- `L_diff/L_latent/L_app/L_edge/L_seg`
- `L1/PSNR/SSIM/LPIPS`

## 9.2 测试集（123 对）
固定 checkpoint 后汇报：
- 平均指标 + 可视化样例
- 重点人工检查：车道线、车辆边界、建筑轮廓是否漂移

---

## 10. 代码改动清单（按文件）

- `dataset.py`
  - 新增 `PairedWeatherDataset`
  - 停用主流程对 `identity_ratio/is_identity` 依赖

- `drivefit_models.py`
  - 新增 `SourceEncoder/StructureEncoder/ConditionFusion`
  - DiT block 增加 cross-attn 条件注入
  - forward 签名改为条件 token 形式
  - 停用 8通道输入分支

- `train.py`
  - dataloader 切换到 paired split
  - 主输入改为 `z_s_noisy`
  - 新增先验提取与多损失计算
  - 增加 stage1/stage2 训练开关和分组学习率

- `sample_edit.py`
  - 与训练同分布推理链路
  - 修复旧接口调用，统一到新条件

- `inversion/base_inverter.py`
  - 编辑接口参数标准化

- `inversion/ddim_inverter.py`
  - 采样函数改用新 forward 条件，移除 8通道拼接

- `requirements.txt`
  - 补充依赖

- `sh/train.sh`
  - 更新为 paired 编辑训练入口

---

## 11. 实施顺序（建议按 PR/提交批次）

1. **PR-1 数据与评估基线**
- split 文件生成 + `PairedWeatherDataset`
- val/test 指标脚手架（先不改模型）

2. **PR-2 训练主输入重构**
- train 主输入改为 `z_s_noisy`
- 去掉 8通道拼接依赖
- 保留最小损失：`L_diff + L_latent`

3. **PR-3 条件模块注入**
- `SourceEncoder + StructureEncoder + cross-attn`
- 跑通 stage1

4. **PR-4 完整损失与推理统一**
- `L_app/L_edge/L_seg` 全开
- 修复 `sample_edit.py + inversion`
- 跑 stage2 与 test

---

## 12. 风险点与回退策略

### 风险 1：多损失初期不稳定
- 回退：先跑 stage1；stage2 逐步增大 `lambda_seg/lambda_edge`

### 风险 2：Seg 先验噪声过大
- 回退：先仅用 `L_edge`，`L_seg` 权重置小（如 0.1）

### 风险 3：模型改动过大导致训练崩
- 回退：只在最后若干 block 注入 cross-attn

---

## 13. 验收标准（第一版）

满足以下即通过：
1. 在 test paired 上，`LPIPS/L1` 相比当前基线下降
2. 可视化中雨天感明显提升
3. 车道线与车辆边界漂移明显减少
4. 训练与推理接口完全一致，无旧 `y`/8通道残留路径

---

## 14. 你确认后我将执行的落地范围

确认本方案后，下一步我会按本文件逐项改代码，默认执行：
- 先做 PR-1 + PR-2（最小可跑）
- 你验收中间结果后，再做 PR-3 + PR-4

