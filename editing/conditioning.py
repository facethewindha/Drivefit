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
# 这两段代码组成了一个很典型的图像条件注入模块：

# 先用卷积把条件图编码成 token
# 再用 cross-attention 把这些 token 注入到主干特征里
# 从而让模型在生成时既保留主干语义，又受到条件图的控制

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
        self.str_encoder = ConvTokenEncoder(in_ch=seg_channels + 1, d_model=d_model) # 这个 encoder 负责提取更偏“结构”的条件：
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

    def _forward_with_context(self, x, t, y_tgt, ctx_tokens):
        """Internal forward given precomputed context tokens."""
        h = self.backbone.x_embedder(x) + self.backbone.pos_embed

        c = self.backbone.t_embedder(t)
        if y_tgt is not None and getattr(self.backbone, "scenario_num", 0) > 0:
            c = c + self.backbone.tgt_weather_embedder(y_tgt, self.training)

        for i, block in enumerate(self.backbone.blocks):
            h = block(h, c)
            if i in self.inject_layers:
                h = self.adapters[str(i)](h, ctx_tokens)

        out = self.backbone.final_layer(h, c)
        out = self.backbone.unpatchify(out)
        return out

    def forward(self, x, t, y_tgt, src_img, seg_logits, edge_map):
        ctx = self.build_context_tokens(src_img, seg_logits, edge_map)
        return self._forward_with_context(x, t, y_tgt, ctx)

    def forward_with_cfg(self, x, t, y_tgt, src_img, seg_logits, edge_map, cfg_scale=4.0):
        """
        Classifier-free guidance for WeatherEditWrapper.
        Unconditional branch:
        - weather label -> null class (scenario_num)
        - context tokens -> zeros
        """
        bsz = x.shape[0]
        device = x.device

        if y_tgt is None:
            return self.forward(x, t, y_tgt, src_img, seg_logits, edge_map)

        if not isinstance(y_tgt, torch.Tensor):
            y_tgt = torch.tensor(y_tgt, device=device, dtype=torch.long)
        y_tgt = y_tgt.to(device).long()
        if y_tgt.ndim == 0:
            y_tgt = y_tgt.view(1).repeat(bsz)
        if y_tgt.shape[0] != bsz:
            y_tgt = y_tgt[:1].repeat(bsz)

        if t.shape[0] != bsz:
            t = t[:1].repeat(bsz)

        ctx_cond = self.build_context_tokens(src_img, seg_logits, edge_map)

        null_idx = getattr(self.backbone, "scenario_num", 0)
        y_uncond = torch.full_like(y_tgt, null_idx)
        ctx_uncond = torch.zeros_like(ctx_cond)

        x_cat = torch.cat([x, x], dim=0)
        t_cat = torch.cat([t, t], dim=0)
        y_cat = torch.cat([y_tgt, y_uncond], dim=0)
        ctx_cat = torch.cat([ctx_cond, ctx_uncond], dim=0)

        model_out = self._forward_with_context(x_cat, t_cat, y_cat, ctx_cat)
        cond_out, uncond_out = torch.split(model_out, bsz, dim=0)

        c = getattr(self.backbone, "in_channels", 4)
        cond_eps, cond_rest = cond_out[:, :c], cond_out[:, c:]
        uncond_eps = uncond_out[:, :c]
        guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        return torch.cat([guided_eps, cond_rest], dim=1)
# 输入
# x：带噪的晴天 latent
# t：扩散步
# y_tgt：目标天气标签
# src_img：原始晴天 RGB 图
# seg_logits：场景分割图
# edge_map：边缘图
# 主干分支
# x -> x_embedder + pos_embed -> h
# 条件向量分支
# t -> t_embedder
# y_tgt -> weather_embedder
# 合成 block condition c
# 条件 token 分支
# src_img -> src_encoder -> src_tok
# [seg_logits, edge_map] -> str_encoder -> str_tok
# [src_tok, str_tok] -> ctx_fuse -> ctx
# 主干迭代

# 对每个 block：

# h = block(h, c)
# 若当前层在注入列表里：
# h = CrossAttentionAdapter(h, ctx)
# 输出
# h -> final_layer -> unpatchify -> out
