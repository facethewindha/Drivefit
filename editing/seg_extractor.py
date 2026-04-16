import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class FrozenSegFormer(nn.Module):
    """
    冻结语义分割先验提取器。
    输入: x in [-1, 1], shape Bx3xHxW
    输出: seg logits, shape BxCxhxw（默认会 resize 到输入大小）
    """
    def __init__(
        self,
        model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
        out_size=(256, 256),
    ):
        super().__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.out_size = out_size

        # ImageNet normalization for segformer
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x):
        # x: [-1,1] -> [0,1]
        x01 = (x + 1.0) / 2.0
        x01 = x01.clamp(0.0, 1.0)
        x_norm = (x01 - self.mean) / self.std

        out = self.model(pixel_values=x_norm)
        logits = out.logits  # B,C,h,w
        logits = F.interpolate(logits, size=self.out_size, mode="bilinear", align_corners=False)
        return logits