import tempfile
import shutil
import numpy as np
from pathlib import Path
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import get_activations, compute_statistics_of_path
import torch
from scipy import linalg

def calculate_fid_stable(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """更稳定的 FID 计算，处理数值问题"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2

    # 添加小的正则化项使矩阵更稳定
    sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
    sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # 处理复数问题
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(f"Warning: Imaginary component detected, using real part only")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid

def main():
    real_path = "./datasets/Ithaca365/Ithaca365-scenario"
    generated_path = "./generated_samples"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 Inception 模型
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        count = 0
        # Windows大小写不敏感，只用小写避免重复
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in Path(real_path).rglob(ext):
                shutil.copy(img_path, f"{temp_dir}/{count:06d}.png")
                count += 1
        print(f"Found {count} real images")
        
        # 计算真实图片统计量
        m1, s1 = compute_statistics_of_path(temp_dir, model, 32, 2048, device, 0)
        # 计算生成图片统计量  
        m2, s2 = compute_statistics_of_path(generated_path, model, 32, 2048, device, 0)
        
        # 使用稳定的 FID 计算
        fid = calculate_fid_stable(m1, s1, m2, s2)
        print(f"FID: {fid}")

if __name__ == '__main__':
    main()