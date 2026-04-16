"""Standard DDIM inversion implementation for conditional weather editing."""

import torch
from tqdm import tqdm

from .base_inverter import BaseInverter


class DDIMInverter(BaseInverter):
    """
    DDIM inversion:
      forward: x0 -> xt (deterministic, eta=0)
      reverse: xt -> x0 (conditional denoise)
    """

    def invert(self, x0_latent, num_steps=50, model_kwargs=None):
        """DDIM forward (inversion): x0 -> xt."""
        if model_kwargs is None:
            model_kwargs = {}

        diff = self._get_base_diffusion()
        step_size = max(1, diff.num_timesteps // int(num_steps))
        timesteps = list(range(0, diff.num_timesteps, step_size))[: int(num_steps)]

        img = x0_latent
        intermediates = [img.clone()]

        self.model.eval()
        with torch.no_grad():
            for i in tqdm(timesteps, desc="DDIM Inversion"):
                t = torch.full((img.shape[0],), i, device=self.device, dtype=torch.long)

                def model_fn(x, tt, **kwargs):
                    return self.model(x, tt, **kwargs)

                out = diff.ddim_reverse_sample(
                    model_fn,
                    img,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                )
                img = out["sample"]
                intermediates.append(img.clone())

        return img, intermediates

    def denoise(self, xt, num_steps=50, model_kwargs=None, cfg_scale=1.0):
        """DDIM reverse (denoise): xt -> x0 under provided conditions."""
        if model_kwargs is None:
            model_kwargs = {}

        cfg_scale = float(cfg_scale) if cfg_scale is not None else 1.0
        use_cfg = cfg_scale > 1.0

        diff = self._get_base_diffusion()
        step_size = max(1, diff.num_timesteps // int(num_steps))
        timesteps = list(range(0, diff.num_timesteps, step_size))[: int(num_steps)][::-1]

        img = xt
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(timesteps, desc="DDIM Denoise"):
                t = torch.full((img.shape[0],), i, device=self.device, dtype=torch.long)

                def model_fn(x, tt, **kwargs):
                    if use_cfg:
                        m = self.model.module if hasattr(self.model, "module") else self.model
                        if hasattr(m, "forward_with_cfg"):
                            return m.forward_with_cfg(x, tt, cfg_scale=cfg_scale, **kwargs)
                    return self.model(x, tt, **kwargs)

                out = diff.ddim_sample(
                    model_fn,
                    img,
                    t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    eta=0.0,
                )
                img = out["sample"]

        return img

    def _get_base_diffusion(self):
        """Get underlying GaussianDiffusion object."""
        if hasattr(self.diffusion, "base_diffusion"):
            return self.diffusion.base_diffusion
        return self.diffusion

    def _get_model_fn(self, use_cfg=False):
        """Compatibility helper kept for future extension."""
        if use_cfg:
            m = self.model.module if hasattr(self.model, "module") else self.model
            return m.forward_with_cfg
        return self.model.module if hasattr(self.model, "module") else self.model
