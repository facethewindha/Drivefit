from .base_inverter import BaseInverter
from .ddim_inverter import DDIMInverter
INVERTER_REGISTRY = {
    "ddim": DDIMInverter,
    # 后续扩展:
    # "null_text": NullTextInverter,
    # "edict": EDICTInverter,
    # "ddpm": DDPMInverter,
}
def create_inverter(inversion_type, diffusion, model, vae, device="cuda"):
    """工厂函数：根据类型创建 inverter"""
    if inversion_type not in INVERTER_REGISTRY:
        raise ValueError(f"Unknown inversion type: {inversion_type}. "
                         f"Available: {list(INVERTER_REGISTRY.keys())}")
    return INVERTER_REGISTRY[inversion_type](diffusion, model, vae, device)