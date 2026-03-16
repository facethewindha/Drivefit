from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torch.nn.functional as F
import os
import numpy as np


def load_info(info_file_path):
    """
    directory/
            ├── class_x
            │   ├── xxx.npy
            │   ├── xxy.npy
            │   └── ...
            └── class_y
                ├── 123.npy
                ├── nsdf3.npy
                └── ...
                └── asd932_.npy

    Args:
        info_file_path (_type_): _description_

    Returns:
        boxes info: tensor
    """
    classes = sorted(
        entry.name for entry in os.scandir(info_file_path) if entry.is_dir()
    )
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {info_file_path}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    # 例如: ['cloudy', 'night', 'rainy', 'snowy', 'sunny']
    
    # Step 2: 建立类别名 → 索引的映射
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    # 例如: {'cloudy': 0, 'night': 1, 'rainy': 2, 'snowy': 3, 'sunny': 4}
    
    # Step 3: 遍历每个类别，收集所有.npy文件

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(info_file_path, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = path, class_index, fname
                instances.append(item)

                if target_class not in available_classes:
                    available_classes.add(target_class)

    return instances


class CustomImageFolder(ImageFolder):
    def __init__(self, root, info_file_path, trans_flip=False, bbox_coord_path=None, max_boxes=30,transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.info = load_info(info_file_path)
        self.trans_flip=trans_flip
        self.p=0.5

        # ============ 新增：加载 bbox 坐标 ============
        self.max_boxes = max_boxes
        self.bbox_coord_path = bbox_coord_path
        if bbox_coord_path is not None:
            self.bbox_coords = load_info(bbox_coord_path)  # 加载坐标文件
        else:
            self.bbox_coords = None
        # ============ 新增结束 ============

    def __getitem__(self, index):
        path, target = self.samples[index]

        img = self.loader(path)

        info, target_info, fname = self.info[index]
        box_info = np.load(info)

          # ============ 新增：加载 bbox 坐标 ============
        if self.bbox_coords is not None:
            coord_path, _, _ = self.bbox_coords[index]
            coords = np.load(coord_path)  # (N, 4) 原始坐标
            
            # 归一化到 [0, 1]
            coords = coords.copy()
            coords[:, [0, 2]] /= 32.0  # x 归一化
            coords[:, [1, 3]] /= 32.0  # y 归一化
            
            # Padding 到固定长度
            num_boxes = min(len(coords), self.max_boxes)
            bbox_padded = np.zeros((self.max_boxes, 4), dtype=np.float32)
            bbox_mask = np.zeros(self.max_boxes, dtype=np.float32)
            
            bbox_padded[:num_boxes] = coords[:num_boxes]
            bbox_mask[:num_boxes] = 1.0
        else:
            bbox_padded = np.zeros((self.max_boxes, 4), dtype=np.float32)
            bbox_mask = np.zeros(self.max_boxes, dtype=np.float32)
        # ============ 新增结束 ============

        assert target == target_info

        if self.trans_flip:
            if torch.rand(1) < self.p:
                img = transforms.functional.hflip(img)
                box_info = np.flip(box_info,axis=1).copy()
                 # ============ 新增：同步翻转 bbox 坐标 ============
                if bbox_mask.sum() > 0:
                    # 翻转 x 坐标: x_new = 1 - x_old
                    bbox_padded[:, [0, 2]] = 1.0 - bbox_padded[:, [2, 0]]
                # ============ 新增结束 ============
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, box_info, fname, bbox_padded, bbox_mask
