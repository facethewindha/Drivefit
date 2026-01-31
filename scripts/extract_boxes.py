"""
extract_boxes.py - 提取车辆边界框并生成掩码文件

功能：为 DriveDiTFit 训练生成 object-sensitive loss 所需的掩码
输入：Ithaca365 数据集元数据 (sample.json, object_ann.json)
输出：./datasets/box_info/{天气类型}/{timestamp}.npy
"""

import json
import os
import numpy as np
from tqdm import tqdm

# ========== 配置 ==========
DATAROOT = "D:/Reproduce/DriveFit/DriveDiTFit/datasets/Ithaca365"
VERSION = "v2.2"
SCENARIO_PATH = "D:\Reproduce\DriveFit\DriveDiTFit\datasets\Ithaca365\Ithaca365-scenario"
OUTPUT_PATH = "./datasets/box_info"
WEATHER_LABELS = ['snow', 'rain', 'night', 'sunny', 'cloud']
VEHICLE_LABELS = ['bus', 'car', 'truck']

# 原始尺寸和目标尺寸
ORIGINAL_WIDTH = 1920
ORIGINAL_HEIGHT = 1208
TARGET_SIZE = 256
LATENT_SIZE = 32

# 缩放因子
SCALE_X = ORIGINAL_WIDTH / TARGET_SIZE  # 7.5
SCALE_Y = ORIGINAL_HEIGHT / TARGET_SIZE  # 4.71875
LATENT_SCALE = TARGET_SIZE / LATENT_SIZE  # 8

print("=" * 50)
print("提取车辆边界框并生成掩码")
print("=" * 50)

# ========== 1. 加载数据（绕过 ithaca365 库的问题）==========
print("\n[1/5] 加载数据集元数据...")

sample_path = os.path.join(DATAROOT, VERSION, "sample.json")
object_ann_path = os.path.join(DATAROOT, VERSION, "object_ann.json")

with open(sample_path, 'r', encoding='utf-8') as f:
    sample = json.load(f)
print(f"  ✓ 加载 {len(sample)} 个样本")

with open(object_ann_path, 'r', encoding='utf-8') as f:
    object_ann = json.load(f)
print(f"  ✓ 加载 {len(object_ann)} 个目标标注")

# ========== 2. 建立相机token到时间戳的映射 ==========
print("\n[2/5] 建立相机token到时间戳映射...")

cam_token2time = {}
for s in sample:
    cam_token2time[s['key_camera_token']] = s['timestamp']
print(f"  ✓ 建立 {len(cam_token2time)} 个映射")

# ========== 3. 提取车辆边界框 ==========
print("\n[3/5] 提取车辆边界框...")

object_dict = {}
vehicle_count = 0

for ann in object_ann:
    if ann['class'] not in VEHICLE_LABELS:
        continue
    
    frame_token = ann['sample_data_token']
    if frame_token not in cam_token2time:
        continue
        
    timestamp = str(cam_token2time[frame_token])
    bbox = ann['bbox']  # [x1, y1, x2, y2]
    
    if timestamp not in object_dict:
        object_dict[timestamp] = []
    object_dict[timestamp].append(bbox)
    vehicle_count += 1

print(f"  ✓ 提取 {vehicle_count} 个车辆框，覆盖 {len(object_dict)} 帧")

# ========== 4. 缩放边界框坐标 ==========
print("\n[4/5] 缩放边界框坐标...")
print(f"  原始尺寸: {ORIGINAL_WIDTH}×{ORIGINAL_HEIGHT}")
print(f"  目标尺寸: {TARGET_SIZE}×{TARGET_SIZE}")
print(f"  Latent尺寸: {LATENT_SIZE}×{LATENT_SIZE}")

for timestamp, boxes in object_dict.items():
    for i in range(len(boxes)):
        # 原始 -> 256x256 -> 32x32
        boxes[i] = [
            boxes[i][0] / SCALE_X / LATENT_SCALE,  # x1
            boxes[i][1] / SCALE_Y / LATENT_SCALE,  # y1
            boxes[i][2] / SCALE_X / LATENT_SCALE,  # x2
            boxes[i][3] / SCALE_Y / LATENT_SCALE,  # y2
        ]

print(f"  ✓ 缩放完成")

# ========== 5. 生成并保存掩码 ==========
print("\n[5/5] 生成掩码文件...")

total_masks = 0
for weather in WEATHER_LABELS:
    save_dir = os.path.join(OUTPUT_PATH, weather)
    os.makedirs(save_dir, exist_ok=True)
    
    scenario_dir = os.path.join(SCENARIO_PATH, weather)
    if not os.path.exists(scenario_dir):
        print(f"  ⚠ 目录不存在: {scenario_dir}")
        continue
    
    files = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.png')])
    
    if not files:
        print(f"  ⚠ {weather}: 没有图片文件")
        continue
    
    weather_count = 0
    for filename in files:
        name = filename.split('.')[0]  # 去掉扩展名（即时间戳）
        
        # 创建32x32的掩码
        mask = np.zeros((LATENT_SIZE, LATENT_SIZE), dtype=np.float32)
        
        if name in object_dict:
            for box in object_dict[name]:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                # 确保框有效
                if x1 + 1 < x2 and y1 + 1 < y2:
                    # 限制在有效范围内
                    x1 = max(0, min(x1, LATENT_SIZE - 1))
                    y1 = max(0, min(y1, LATENT_SIZE - 1))
                    x2 = max(0, min(x2, LATENT_SIZE))
                    y2 = max(0, min(y2, LATENT_SIZE))
                    mask[y1:y2, x1:x2] = 1
        
        np.save(os.path.join(save_dir, f"{name}.npy"), mask)
        weather_count += 1
    
    print(f"  ✓ {weather}: 生成 {weather_count} 个掩码")
    total_masks += weather_count

print("\n" + "=" * 50)
print(f"完成！共生成 {total_masks} 个掩码文件")
print(f"保存位置: {os.path.abspath(OUTPUT_PATH)}")
print("=" * 50)