"""统一 Inversion 接口，为后续接入 improved DDIM / null-text 等预留"""
from abc import ABC, abstractmethod
import torch
class BaseInverter(ABC):
    """
    所有 Inversion 方法的基类。
    后续扩展只需继承此类并实现 invert() 和 denoise()。
    """
    def __init__(self, diffusion, model, vae, device="cuda"):
        self.diffusion = diffusion      # GaussianDiffusion 或 SpacedDiffusion
        self.model = model              # DiT 模型
        self.vae = vae
        self.device = device
    def encode_image(self, img_tensor):
        """RGB tensor [-1,1] → VAE latent"""
        with torch.no_grad():
            return self.vae.encode(img_tensor).latent_dist.sample().mul_(0.18215)
    def decode_latent(self, latent):
        """VAE latent → RGB tensor [0,255]"""
        with torch.no_grad():
            decoded = self.vae.decode(latent / 0.18215).sample
        return torch.clamp(127.5 * decoded + 128.0, 0, 255)
    @abstractmethod
    def invert(self, x0_latent, num_steps, model_kwargs=None):
        """
        将 clean latent x₀ 映射到噪声空间 xₜ。
        Returns:
            xt: 噪声空间的 latent
            intermediates: 中间结果 (可选，用于 improved 方法)
        """
        pass
    @abstractmethod
    def denoise(self, xt, num_steps, model_kwargs=None, cfg_scale=1.0):
        """
        从 xₜ 去噪回 x₀。
        Returns:
            x0_edited: 编辑后的 latent
        """
        pass
    def edit(self, img_tensor, source_weather_label,target_weather_label,
             inversion_steps=50, denoise_steps=50,
             source_model_kwargs=None, target_model_kwargs=None,
             cfg_scale=4.0):
        """
        完整编辑流程: encode → invert → denoise(新条件) → decode
        这是对外的统一调用接口。
        """
        batch_size = img_tensor.shape[0]
        x0 = self.encode_image(img_tensor)
        z_src = x0  # 源图 latent 作为 conditioning
        # 2. 补全 Inversion 参数 (Sunny -> Sunny)
        # 必须提供 y_src 和 y_tgt 否则模型 forward 会报错
        source_model_kwargs = {
            "y_src": torch.full((batch_size,), source_weather_label, device=self.device).long(),
            "y_tgt": torch.full((batch_size,), source_weather_label, device=self.device).long()
        }
        # 3. 补全 Denoise 参数 (Sunny -> Rain)
        target_model_kwargs = {
            "y_src": torch.full((batch_size,), source_weather_label, device=self.device).long(),
            "y_tgt": torch.full((batch_size,), target_weather_label, device=self.device).long()
        }
        xt, intermediates = self.invert(x0,z_src, inversion_steps, source_model_kwargs)
        x0_edited = self.denoise(xt,z_src, denoise_steps, target_model_kwargs, cfg_scale)
        return self.decode_latent(x0_edited), intermediates