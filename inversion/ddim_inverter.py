"""标准 DDIM Inversion 实现，复用项目已有的 ddim_reverse_sample"""
import torch
from tqdm import tqdm
from .base_inverter import BaseInverter
class DDIMInverter(BaseInverter):
    """
    标准 DDIM Inversion:
      forward: x₀ → x₁ → x₂ → ... → xₜ  (确定性，eta=0)
      reverse: xₜ → x_{t-1} → ... → x₀   (DDIM denoise)
    
    复用 GaussianDiffusion.ddim_reverse_sample (L664-703)
    """
    def invert(self, x0_latent, z_src, num_steps=50, model_kwargs=None):
        """DDIM forward: x0 → xt，每步输入拼入 z_src"""
        """
        DDIM forward (inversion): x₀ → xₜ
        利用 diffusion.ddim_reverse_sample 逐步正向扩散
        """
        if model_kwargs is None:
            model_kwargs = {}
        # 获取 base diffusion (原始1000步的)
        diff = self._get_base_diffusion()
        
        # 计算要使用的时间步 (均匀采样 num_steps 个)
        step_size = diff.num_timesteps // num_steps
        timesteps = list(range(0, diff.num_timesteps, step_size))[:num_steps]
        img = x0_latent
        intermediates = [img.clone()]
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(timesteps, desc="DDIM Inversion"):
                t = torch.tensor([i] * img.shape[0], device=self.device)
                # --- Wrapper: 在模型调用的瞬间拼入静态的 z_src ---
                def model_fn(x, t, **kwargs):
                    x_in = torch.cat([x, z_src], dim=1)
                    return self.model(x_in, t, **kwargs)

                x_input = torch.cat([img, z_src], dim=1)
                out = diff.ddim_reverse_sample(
                    model_fn,
                    img, t,# 传给扩散引擎的是 4 通道逻辑
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                )
                img = out["sample"]
                intermediates.append(img.clone())
        return img, intermediates
    def denoise(self, xt, z_src, num_steps=50, model_kwargs=None, cfg_scale=1.0):
        """DDIM reverse: xt → x0，每步输入拼入 z_src"""
        """
        DDIM reverse (denoise): xₜ → x₀，用目标天气条件引导
        """
        if model_kwargs is None:
            model_kwargs = {}
        diff = self._get_base_diffusion()
        step_size = diff.num_timesteps // num_steps
        timesteps = list(range(0, diff.num_timesteps, step_size))[:num_steps]
        timesteps = timesteps[::-1]  # 逆序：从大到小
        use_cfg = cfg_scale > 1.0
        img = xt
        if use_cfg:
            img = torch.cat([img, img], dim=0)
            y = model_kwargs["y_tgt"]
            # 获取 null class index
            m = self.model.module if hasattr(self.model, "module") else self.model
            null_idx = m.scenario_num
            y_null = torch.tensor([null_idx] * y.shape[0],device=self.device)
            model_kwargs["y_tgt"] = torch.cat([y, y_null], dim=0)
            y_src = model_kwargs["y_src"]
            model_kwargs["y_src"] = torch.cat([y_src, y_src], dim=0)
            # 同步把 z_src 拼成 2 倍
            z_src_cfg = torch.cat([z_src, z_src], dim=0)
            # model_kwargs 需要包含 cfg_scale
            model_kwargs["cfg_scale"] = cfg_scale
        else:
            z_src_cfg = z_src
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(timesteps, desc="DDIM Denoise"):
                 # 采样器的 t 维度必须匹配当前 img (B 或 2B)
                actual_b = img.shape[0]
                t = torch.tensor([i] * actual_b, device=self.device)
                # --- 核心 Wrapper：内部处理 8 通道拼接和 CFG 逻辑 ---
                def model_fn(x, t, **kwargs):
                    x_in = torch.cat([x, z_src_cfg], dim=1) # 拼成 8 通道
                    if use_cfg:
                        # 指向你在 drivefit_models.py 中定义的 forward_with_cfg
                        return self.model.module.forward_with_cfg(x_in, t, **kwargs) if hasattr(self.model, 'module') else self.model.forward_with_cfg(x_in, t, **kwargs)
                    else:
                        return self.model.module(x_in, t, **kwargs) if hasattr(self.model, 'module') else self.model(x_in, t, **kwargs)
                out = diff.ddim_sample(
                    model_fn,
                    img, t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                )
                img = out["sample"]
        if use_cfg:
            img, _ = img.chunk(2, dim=0)
        return img
    def _get_base_diffusion(self):
        """获取底层 GaussianDiffusion 对象"""
        if hasattr(self.diffusion, 'base_diffusion'):
            return self.diffusion.base_diffusion
        return self.diffusion
    def _get_model_fn(self, use_cfg=False):
        """获取模型前向函数"""
        if use_cfg:
            m = self.model.module if hasattr(self.model, 'module') else self.model
            return m.forward_with_cfg
        return self.model.module if hasattr(self.model, 'module') else self.model