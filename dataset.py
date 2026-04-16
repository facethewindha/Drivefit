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
    def __init__(self, root, info_file_path, trans_flip=False, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.info = load_info(info_file_path)
        self.trans_flip=trans_flip
        self.p=0.5

    def __getitem__(self, index):
        path, target = self.samples[index]

        img = self.loader(path)

        info, target_info, fname = self.info[index]
        box_info = np.load(info)
        assert target == target_info

        if self.trans_flip:
            if torch.rand(1) < self.p:
                img = transforms.functional.hflip(img)
                box_info = np.flip(box_info,axis=1).copy()
        
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, box_info, fname

# class WeatherEditDataset(Dataset):
#     """
#     天气编辑数据集。数据组织:
#         data_root/sunny/xxx.png
#         data_root/rain/xxx.png
#     """
#     WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}

#     def __init__(self, data_root, source_weather="sunny", target_weather="rain",
#                  transform=None,identity_ratio=0.4,
#                  split="train", split_ratio=0.8, seed=42):
#         super().__init__()
#         self.data_root = data_root
#         self.source_weather = source_weather
#         self.target_weather = target_weather
#         self.transform = transform
#         import random
#         rng = random.Random(seed)

#         source_dir = os.path.join(data_root, source_weather)
#         self.source_imgs = sorted([
#             os.path.join(source_dir, f) for f in os.listdir(source_dir)
#             if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#         ])
        
#         target_dir = os.path.join(data_root, target_weather)
#         self.target_imgs = sorted([
#             os.path.join(target_dir, f) for f in os.listdir(target_dir)
#             if f.lower().endswith(('.png', '.jpg', '.jpeg'))
#         ]) if os.path.isdir(target_dir) else []
        
#         n = len(self.source_imgs)
#         indices = list(range(n))
#         rng.shuffle(indices)
#         split_idx = int(n * split_ratio)
#         if split == "train":
#             selected = indices[:split_idx]
#         else:  # test
#             selected = indices[split_idx:]
#         self.source_imgs = [self.source_imgs[i] for i in sorted(selected)]

#         self.target_label = self.WEATHER_TO_IDX[target_weather]

#     def __len__(self):
#         return len(self.source_imgs)

#     def __getitem__(self, index):
#         from PIL import Image as PILImage
#         source_img = PILImage.open(self.source_imgs[index]).convert("RGB")
#         if self.transform:
#             source_img = self.transform(source_img)
#         result = {"source_img": source_img, "target_weather_label": self.target_label}
#         if self.return_ref and self.target_imgs:
#             ref_idx = torch.randint(0, len(self.target_imgs), (1,)).item()
#             ref_img = PILImage.open(self.target_imgs[ref_idx]).convert("RGB")
#             if self.transform:
#                 ref_img = self.transform(ref_img)
#             result["target_ref_img"] = ref_img
#         return result


class WeatherEditDataset(Dataset):
    """
    非配对天气编辑数据集。数据组织:
        data_root/sunny/xxx.png
        data_root/rain/xxx.png
    返回: src_img, tgt_img, src_label, tgt_label, is_identity
    """
    WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}

    def __init__(self, data_root, source_weather="sunny", target_weather="rain",
                 transform=None, identity_ratio=0.4,
                 split="train", split_ratio=0.8, seed=42):
        super().__init__()
        self.data_root = data_root
        self.source_weather = source_weather
        self.target_weather = target_weather
        self.transform = transform
        self.identity_ratio = identity_ratio  # identity 样本占比

        import random
        rng = random.Random(seed)

        # 加载 source 图（sunny）
        source_dir = os.path.join(data_root, source_weather)
        all_src = sorted([
            os.path.join(source_dir, f) for f in os.listdir(source_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        # 按 split_ratio 划分
        n = len(all_src)
        idx = list(range(n))
        rng.shuffle(idx)
        cut = int(n * split_ratio)
        self.source_imgs = [all_src[i] for i in sorted(idx[:cut] if split=="train" else idx[cut:])]

        # 加载 target 图（rain），同样划分
        target_dir = os.path.join(data_root, target_weather)
        if os.path.isdir(target_dir):
            all_tgt = sorted([
                os.path.join(target_dir, f) for f in os.listdir(target_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            m = len(all_tgt)
            idx2 = list(range(m))
            rng.shuffle(idx2)
            cut2 = int(m * split_ratio)
            self.target_imgs = [all_tgt[i] for i in sorted(idx2[:cut2] if split=="train" else idx2[cut2:])]
        else:
            self.target_imgs = []

        self.src_label = self.WEATHER_TO_IDX[source_weather]
        self.tgt_label = self.WEATHER_TO_IDX[target_weather]

    def __len__(self):
        return len(self.source_imgs)

    def __getitem__(self, index):
        import random
        from PIL import Image as PILImage

        # 先固定拿 source 图，保证 src_img 一定存在
        src_path = self.source_imgs[index]
        src_img = PILImage.open(src_path).convert("RGB")
        if self.transform:
            src_img = self.transform(src_img)

        is_identity = random.random() < self.identity_ratio

        if is_identity:
            # identity 样本：要么用 source 图做 identity，要么随机用 target 图做 identity
            use_source = random.random() < 0.5

            if use_source:
                img = src_img
                label = self.src_label
            else:
                idx_tgt = random.randint(0, len(self.target_imgs) - 1)
                img = PILImage.open(self.target_imgs[idx_tgt]).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                label = self.tgt_label

            src_img = img
            tgt_img = img
            src_label = label
            tgt_label = label

        else:
        # editing 样本：src 固定是 source_imgs[index]
            if self.target_imgs:
                tgt_idx = random.randint(0, len(self.target_imgs) - 1)
                tgt_img = PILImage.open(self.target_imgs[tgt_idx]).convert("RGB")
                if self.transform:
                    tgt_img = self.transform(tgt_img)
            else:
                tgt_img = src_img  # 没有 target 图时退化为 identity

            src_label = self.src_label
            tgt_label = self.tgt_label

        return {
            "src_img": src_img,
            "tgt_img": tgt_img,
            "src_label": src_label,
            "tgt_label": tgt_label,
            "is_identity": int(is_identity),
        }


class PairedWeatherDataset(Dataset):
    """
    按文件排序后按索引一一配对:
      source_imgs[i] <-> target_imgs[i]
    仅使用前 max_pairs 对（默认 1225 对）。
    """

    WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}

    def __init__(
        self,
        data_root,
        source_weather="sunny",
        target_weather="rain",
        transform=None,
        max_pairs=1225,
        split="train",
        train_count=980,
        val_count=122,
        test_count=123,
    ):
        super().__init__()
        self.data_root = data_root
        self.source_weather = source_weather
        self.target_weather = target_weather
        self.transform = transform

        source_dir = os.path.join(data_root, source_weather)
        target_dir = os.path.join(data_root, target_weather)

        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Source weather dir not found: {source_dir}")
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Target weather dir not found: {target_dir}")

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        source_imgs = sorted(
            [
                os.path.join(source_dir, f)
                for f in os.listdir(source_dir)
                if f.lower().endswith(valid_exts)
            ]
        )
        target_imgs = sorted(
            [
                os.path.join(target_dir, f)
                for f in os.listdir(target_dir)
                if f.lower().endswith(valid_exts)
            ]
        )

        pair_num = min(len(source_imgs), len(target_imgs), int(max_pairs))
        if pair_num <= 0:
            raise RuntimeError(
                f"No valid pairs found. source={len(source_imgs)}, target={len(target_imgs)}"
            )

        source_imgs = source_imgs[:pair_num]
        target_imgs = target_imgs[:pair_num]

        total_need = int(train_count) + int(val_count) + int(test_count)
        if total_need > pair_num:
            raise ValueError(
                f"Requested split size {total_need} exceeds available pair num {pair_num}"
            )

        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"Unsupported split: {split}")

        if split == "train":
            s, e = 0, int(train_count)
        elif split == "val":
            s, e = int(train_count), int(train_count) + int(val_count)
        elif split == "test":
            s = int(train_count) + int(val_count)
            e = s + int(test_count)
        else:  # all
            s, e = 0, pair_num

        self.source_imgs = source_imgs[s:e]
        self.target_imgs = target_imgs[s:e]
        self.src_label = self.WEATHER_TO_IDX[source_weather]
        self.tgt_label = self.WEATHER_TO_IDX[target_weather]

    def __len__(self):
        return len(self.source_imgs)

    def __getitem__(self, index):
        from PIL import Image as PILImage

        src_img = PILImage.open(self.source_imgs[index]).convert("RGB")
        tgt_img = PILImage.open(self.target_imgs[index]).convert("RGB")

        if self.transform is not None:
            src_img = self.transform(src_img)
            tgt_img = self.transform(tgt_img)

        return {
            "src_img": src_img,
            "tgt_img": tgt_img,
            "src_label": self.src_label,
            "tgt_label": self.tgt_label,
            "pair_index": index,
        }


class PairedWeatherReconDataset(Dataset):
    """
    Stage1 reconstruction dataset built from paired data, but expanded to single-weather items.
    For each pair i, create two samples:
      2*i   -> source weather image (e.g. sunny)
      2*i+1 -> target weather image (e.g. rain)
    """

    WEATHER_TO_IDX = {"sunny": 0, "rain": 1, "snow": 2, "cloud": 3, "night": 4}

    def __init__(
        self,
        data_root,
        source_weather="sunny",
        target_weather="rain",
        transform=None,
        max_pairs=1225,
    ):
        super().__init__()
        self.data_root = data_root
        self.source_weather = source_weather
        self.target_weather = target_weather
        self.transform = transform

        source_dir = os.path.join(data_root, source_weather)
        target_dir = os.path.join(data_root, target_weather)
        if not os.path.isdir(source_dir):
            raise FileNotFoundError(f"Source weather dir not found: {source_dir}")
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Target weather dir not found: {target_dir}")

        valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        source_imgs = sorted(
            [
                os.path.join(source_dir, f)
                for f in os.listdir(source_dir)
                if f.lower().endswith(valid_exts)
            ]
        )
        target_imgs = sorted(
            [
                os.path.join(target_dir, f)
                for f in os.listdir(target_dir)
                if f.lower().endswith(valid_exts)
            ]
        )

        pair_num = min(len(source_imgs), len(target_imgs), int(max_pairs))
        if pair_num <= 0:
            raise RuntimeError(
                f"No valid pairs found. source={len(source_imgs)}, target={len(target_imgs)}"
            )

        self.source_imgs = source_imgs[:pair_num]
        self.target_imgs = target_imgs[:pair_num]
        self.src_label = self.WEATHER_TO_IDX[source_weather]
        self.tgt_label = self.WEATHER_TO_IDX[target_weather]

    def __len__(self):
        return 2 * len(self.source_imgs)

    def __getitem__(self, index):
        from PIL import Image as PILImage

        pair_idx = index // 2
        is_source = (index % 2 == 0)

        if is_source:
            img_path = self.source_imgs[pair_idx]
            label = self.src_label
        else:
            img_path = self.target_imgs[pair_idx]
            label = self.tgt_label

        img = PILImage.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return {
            "img": img,
            "label": label,
            "pair_index": pair_idx,
            "is_source": int(is_source),
        }
