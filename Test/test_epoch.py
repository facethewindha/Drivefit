import torch

ckpt = torch.load("./results/ithaca365/001-DiT-XL-2/checkpoints/0017500.pt", map_location="cpu")
print("Keys:", ckpt.keys())
print("Epoch:", ckpt.get("epoch", "未保存"))
print("Train steps:", ckpt.get("train_steps", "未保存"))