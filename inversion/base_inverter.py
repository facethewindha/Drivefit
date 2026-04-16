"""Unified inverter interface for DDIM-based editing."""
from abc import ABC, abstractmethod
import torch
class BaseInverter(ABC):
    """
    Base class for all inversion methods.
    Subclasses only need to implement `invert()` and `denoise()`.
    """
    def __init__(self, diffusion, model, vae, device="cuda"):
        self.diffusion = diffusion      # GaussianDiffusion or SpacedDiffusion
        self.model = model              # DiT / wrapper model
        self.vae = vae
        self.device = device
    def encode_image(self, img_tensor):
        """RGB tensor [-1,1] -> VAE latent."""
        with torch.no_grad():
            return self.vae.encode(img_tensor).latent_dist.sample().mul_(0.18215)
    def decode_latent(self, latent):
        """VAE latent -> RGB tensor [0,255]."""
        with torch.no_grad():
            decoded = self.vae.decode(latent / 0.18215).sample
        return torch.clamp(127.5 * decoded + 128.0, 0, 255)
    @abstractmethod
    def invert(self, x0_latent, num_steps, model_kwargs=None):
        """
        Map clean latent x0 to noisy latent xt.
        Returns:
            xt: noisy latent
            intermediates: optional intermediate states
        """
        pass
    @abstractmethod
    def denoise(self, xt, num_steps, model_kwargs=None, cfg_scale=1.0):
        """
        Denoise xt back to x0 under target condition.
        Returns:
            x0_edited: edited latent
        """
        pass
    def edit(
        self,
        img_tensor,
        source_weather_label,
        target_weather_label,
        src_img,
        seg_logits,
        edge_map,
        inversion_steps=50,
        denoise_steps=50,
        cfg_scale=4.0,
    ):
        """
        DDIM inversion + conditional denoise
        """
        batch_size = img_tensor.shape[0]
        x0 = self.encode_image(img_tensor)

        if isinstance(source_weather_label, int):
            y_src = torch.full((batch_size,), source_weather_label, device=self.device).long()
        else:
            y_src = source_weather_label.to(self.device).long()

        if isinstance(target_weather_label, int):
            y_tgt = torch.full((batch_size,), target_weather_label, device=self.device).long()
        else:
            y_tgt = target_weather_label.to(self.device).long()

        source_model_kwargs = {
            "y_tgt": y_src,
            "src_img": src_img,
            "seg_logits": seg_logits,
            "edge_map": edge_map,
        }
        target_model_kwargs = {
            "y_tgt": y_tgt,
            "src_img": src_img,
            "seg_logits": seg_logits,
            "edge_map": edge_map,
        }

        xt, intermediates = self.invert(x0, num_steps=inversion_steps, model_kwargs=source_model_kwargs)
        x0_edited = self.denoise(xt, num_steps=denoise_steps, model_kwargs=target_model_kwargs, cfg_scale=cfg_scale)
        return self.decode_latent(x0_edited), intermediates
