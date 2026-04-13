"""编辑损失函数 —— 预留接口，不阻塞主流程"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptualConsistencyLoss(nn.Module):
    """感知一致性损失 (latent MSE 简版，后续可改为 VGG feature matching)"""
    def forward(self, source_latent, edited_latent):
        return F.mse_loss(edited_latent, source_latent)

class EdgeConsistencyLoss(nn.Module):
    """边缘一致性损失 (Sobel 边缘比较)"""
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).reshape(1,1,3,3)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).reshape(1,1,3,3)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)
    
    def get_edges(self, x):
        B, C, H, W = x.shape
        edges = []
        for c in range(C):
            xc = x[:, c:c+1]
            edges.append(torch.sqrt(
                F.conv2d(xc, self.sx.to(x.device), padding=1)**2 +
                F.conv2d(xc, self.sy.to(x.device), padding=1)**2 + 1e-8))
        return torch.cat(edges, dim=1)
    
    def forward(self, source_latent, edited_latent):
        return F.mse_loss(self.get_edges(edited_latent), self.get_edges(source_latent))

class WeatherClassifierLoss(nn.Module):
    """天气分类损失 (预留，需要预训练分类器)"""
    def forward(self, edited_img, target_label):
        return torch.tensor(0.0, device=edited_img.device, requires_grad=True)