import os
import re
import cv2
import torch
import numpy as np
from config import config
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, Optional, List
import albumentations as A
from albumentations.pytorch import ToTensorV2


_SEQ_PATTERN = re.compile(r'_(t2|c2|c5)_slice_', re.IGNORECASE)

def _parse_seq_from_filename(path: str):
    """
    从文件名中解析序列类型：t2 / c2 / c5
    若解析失败，返回 None
    """
    fname = os.path.basename(path)
    m = _SEQ_PATTERN.search(fname)
    if m is None:
        return None
    return m.group(1).lower()


def _filter_pairs_by_seq_mode(image_paths, mask_paths, seq_mode: str):
    """
    根据 seq_mode 过滤 image/mask 对
    image_paths, mask_paths: List[str]
    返回过滤后的 (image_paths, mask_paths)
    """
    if seq_mode is None or seq_mode.upper() == "ALL":
        return image_paths, mask_paths

    seq_mode = seq_mode.upper()

    # seq_mode -> 实际文件序列名
    if seq_mode == "T2":
        allow = {"t2"}
    elif seq_mode == "C2":
        allow = {"c2"}
    elif seq_mode == "C5":
        allow = {"c5"}
    elif seq_mode in {"C2+C5", "C5+C2"}:
        allow = {"c2", "c5"}
    else:
        raise ValueError(f"[SeqMode] 不支持的序列模式: {seq_mode}")

    filtered_imgs = []
    filtered_masks = []

    for img, msk in zip(image_paths, mask_paths):
        seq = _parse_seq_from_filename(img)
        if seq is None:
            # 极端情况：文件名异常，直接跳过，避免污染训练
            continue
        if seq in allow:
            filtered_imgs.append(img)
            filtered_masks.append(msk)

    return filtered_imgs, filtered_masks


class MedicalImageDataset(Dataset):
    """医学图像分割数据集 - 加载PNG格式的图像和掩码"""

    def __init__(self,
                 image_paths: List[str],
                 mask_paths: List[str],
                 transform: Optional[callable] = None,
                 is_train: bool = True):
        """
        初始化数据集

        Args:
            image_paths: 图像文件路径列表
            mask_paths: 掩码文件路径列表
            transform: 数据增强变换
            is_train: 是否为训练模式
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        self.is_train = is_train

        # 验证文件存在性
        self._validate_files()

        print(f"成功加载 {len(self.image_paths)} 个PNG切片文件")

    def _validate_files(self):
        """验证文件存在性"""
        valid_image_paths = []
        valid_mask_paths = []

        for img_path, mask_path in zip(self.image_paths, self.mask_paths):
            if os.path.exists(img_path) and os.path.exists(mask_path):
                # 检查文件是否损坏
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None and mask is not None:
                        valid_image_paths.append(img_path)
                        valid_mask_paths.append(mask_path)
                except:
                    print(f"文件损坏: {img_path} 或 {mask_path}")
                    continue
            else:
                print(f"文件不存在: {img_path} 或 {mask_path}")

        self.image_paths = valid_image_paths
        self.mask_paths = valid_mask_paths

        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError("图像和掩码文件数量不一致")

    def __len__(self) -> int:
        """返回总切片数"""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取指定索引的数据"""
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # 加载PNG图像
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"无法加载图像: {img_path}")
        if mask is None:
            raise ValueError(f"无法加载掩码: {mask_path}")

        # 数据预处理
        image, mask = self._preprocess_slice(image, mask)

        # 数据增强
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # 转换为Tensor - 确保是4D张量 [1, H, W]
            image = torch.from_numpy(image).float().unsqueeze(0)
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        # 确保掩码是二值的
        mask = (mask > 0.5).float()

        return image, mask

    def _preprocess_slice(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """切片预处理"""
        # 归一化
        image = self._normalize_image(image)

        # 调整尺寸（如果需要）
        if image.shape != config.IMAGE_SIZE:
            image = self._resize_image(image, config.IMAGE_SIZE)
            mask = self._resize_image(mask, config.IMAGE_SIZE, is_mask=True)

        return image, mask

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """图像归一化"""
        image = image.astype(np.float32)
        if np.max(image) > 0:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        return image

    def _resize_image(self, image: np.ndarray, target_size: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
        """调整图像尺寸"""
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(image, target_size, interpolation=interpolation)


def get_transforms():
    """针对病灶的数据增强策略"""
    train_transform = A.Compose([
        # 增强病灶对比度
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),

        # 空间变换时保护病灶区域
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(p=0.3),
        ], p=0.5),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def load_file_list(file_path: str) -> Tuple[List[str], List[str]]:
    """从文件列表加载图像和掩码路径"""
    image_paths = []
    mask_paths = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件列表不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                img_path, mask_path = parts
                image_paths.append(img_path)
                mask_paths.append(mask_path)

    return image_paths, mask_paths


def create_data_loaders(distributed: bool = False, rank: int = 0, world_size: int = 1, seq_mode="ALL"):
    """创建数据加载器 - 支持DDP DistributedSampler，并提升DataLoader吞吐"""
    from torch.utils.data.distributed import DistributedSampler

    # 加载各数据集文件列表
    train_list_file = os.path.join(config.PROCESSED_DATA_DIR, "train_list.txt")
    val_list_file = os.path.join(config.PROCESSED_DATA_DIR, "val_list.txt")
    test_list_file = os.path.join(config.PROCESSED_DATA_DIR, "test_list.txt")

    print("正在加载数据集文件列表...")
    seq_mode = seq_mode or getattr(config, "SEQ_MODE", "ALL")
    train_images, train_masks = load_file_list(train_list_file)
    val_images, val_masks = load_file_list(val_list_file)
    test_images, test_masks = load_file_list(test_list_file)

    train_images, train_masks = _filter_pairs_by_seq_mode(
        train_images, train_masks, seq_mode
    )
    val_images, val_masks = _filter_pairs_by_seq_mode(
        val_images, val_masks, seq_mode
    )
    test_images, test_masks = _filter_pairs_by_seq_mode(
        test_images, test_masks, seq_mode
    )

    print(f"[Data] 使用序列模式: {seq_mode} | "
          f"Train={len(train_images)} Val={len(val_images)} Test={len(test_images)}")

    # print(f"数据集加载: 训练集 {len(train_images)}, 验证集 {len(val_images)}, 测试集 {len(test_images)}")

    # 获取数据变换
    train_transform, val_transform = get_transforms()

    # 创建数据集
    train_dataset = MedicalImageDataset(train_images, train_masks, train_transform, is_train=True)
    val_dataset = MedicalImageDataset(val_images, val_masks, val_transform, is_train=False)
    test_dataset = MedicalImageDataset(test_images, test_masks, val_transform, is_train=False)

    # ✅ DataLoader 提速参数
    num_workers = int(getattr(config, "NUM_WORKERS", 4))
    pin_memory = True
    persistent_workers = (num_workers > 0)
    prefetch_factor = int(getattr(config, "PREFETCH_FACTOR", 2))  # 可在config里加
    drop_last = bool(getattr(config, "DROP_LAST", True))

    # ✅ DDP sampler
    train_sampler = None
    val_sampler = None
    test_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=drop_last)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(train_sampler is None),               # ✅ sampler存在就不shuffle
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if persistent_workers else None,
        drop_last=drop_last
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if persistent_workers else None,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor if persistent_workers else None,
        drop_last=False
    )

    return train_loader, val_loader, test_loader, train_sampler



if __name__ == "__main__":
    # 测试数据加载
    train_loader, val_loader, test_loader = create_data_loaders()
    print("数据加载器创建成功!")

    # 显示批次信息
    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"训练集批次 {batch_idx}: 图像形状 {images.shape}, 掩码形状 {masks.shape}")
        if batch_idx >= 2:  # 只显示前3个批次
            break

    for batch_idx, (images, masks) in enumerate(val_loader):
        print(f"验证集批次 {batch_idx}: 图像形状 {images.shape}, 掩码形状 {masks.shape}")
        if batch_idx >= 1:  # 只显示前2个批次
            break
