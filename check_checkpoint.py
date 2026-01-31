import torch

# 检查 checkpoint 文件的内容
checkpoint_path = r"D:\Reproduce\DriveFit\DriveDiTFit\results\ithaca365\001-DiT-XL-2\checkpoints\0010400.pt"

print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu")

print(f"\nCheckpoint type: {type(checkpoint)}")

if isinstance(checkpoint, dict):
    print(f"\nCheckpoint keys: {checkpoint.keys()}")
    
    if "model" in checkpoint:
        print(f"\n✓ Found 'model' key")
        print(f"  Model state_dict has {len(checkpoint['model'])} parameters")
    else:
        print(f"\n✗ No 'model' key found")
    
    if "epoch" in checkpoint:
        print(f"\n✓ Found 'epoch': {checkpoint['epoch']}")
    else:
        print(f"\n✗ No 'epoch' key")
    
    if "train_steps" in checkpoint:
        print(f"\n✓ Found 'train_steps': {checkpoint['train_steps']}")
    else:
        print(f"\n✗ No 'train_steps' key")
else:
    print("\nCheckpoint is a direct state_dict (not a dictionary)")
    print(f"Number of parameters: {len(checkpoint)}")
