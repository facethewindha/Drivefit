"""天气编辑配置"""
from dataclasses import dataclass, field

@dataclass
class WeatherEditConfig:
    task_type: str = "image_editing"
    source_weather: str = "sunny"
    target_weather: str = "rain"
    num_weather_classes: int = 2      # MVP: sunny(0), rain(1)
    
    # 停用开关
    use_scos: bool = False
    use_progressive_tuning: bool = False
    use_ssei: bool = False
    use_object_sensitive_loss: bool = False
    
    # 编辑参数
    edit_timestep: int = 500
    inversion_type: str = "ddim"      # "ddim" | 后续 "null_text" 等
    inversion_steps: int = 50
    denoise_steps: int = 50
    
    # 损失
    use_perceptual_loss: bool = False
    use_edge_loss: bool = False
    use_weather_classifier: bool = False
    perceptual_loss_weight: float = 0.1
    edge_loss_weight: float = 0.05
    
    noise_schedule: str = "linear"