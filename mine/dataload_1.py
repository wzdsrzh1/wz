import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from config import Config


def get_image_paths(image_path, extensions=('.png', '.jpg', '.jpeg')):
    """
    获取图像路径列表，确保排序一致性

    Args:
        image_path: 图像目录路径
        extensions: 支持的图像扩展名

    Returns:
        排序后的图像路径列表
    """
    file_names = [f for f in os.listdir(image_path)
                  if f.lower().endswith(extensions)]
    file_names.sort()  # 确保顺序一致
    img_path_list = [os.path.join(image_path, fname) for fname in file_names]
    return img_path_list


def validate_paths(paths1, paths2, modality1="PET", modality2="MRI"):
    """
    验证两个模态的图像路径是否匹配
    """
    if len(paths1) != len(paths2):
        raise ValueError(f"{modality1}和{modality2}图像数量不匹配: "
                         f"{len(paths1)} vs {len(paths2)}")

    # 可以添加更详细的文件名匹配验证
    for i, (p1, p2) in enumerate(zip(paths1, paths2)):
        # 简单的文件名检查（可选）
        name1 = os.path.basename(p1).split('.')[0]
        name2 = os.path.basename(p2).split('.')[0]
        if name1 != name2:
            print(f"警告: 第{i}个图像可能不匹配: {name1} vs {name2}")

    return True


class FusionDataset(Dataset):
    """医学图像融合数据集"""

    def __init__(self, source1_paths, source2_paths, transform=None, target_size=(256, 256)):
        self.source1_paths = source1_paths
        self.source2_paths = source2_paths
        self.target_size = target_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

        # 验证路径匹配
        validate_paths(source1_paths, source2_paths, "PET", "MRI")

    def __len__(self):
        return len(self.source1_paths)

    def __getitem__(self, idx):
        try:
            # 加载PET图像
            source1 = Image.open(self.source1_paths[idx]).convert('L')  # 转换为灰度
            source1 = self.transform(source1)  # [1, H, W] 或 [C, H, W]

            # 加载MRI图像
            source2 = Image.open(self.source2_paths[idx]).convert('L')
            source2 = self.transform(source2)

            return source1, source2

        except Exception as e:
            print(f"加载图像失败: {self.source1_paths[idx]}, {self.source2_paths[idx]}")
            print(f"错误: {e}")
            # 返回空白图像或重新尝试
            return torch.zeros(1, *self.target_size), torch.zeros(1, *self.target_size)

    def get_image_info(self, idx):
        """获取图像信息（用于调试）"""
        info = {
            'pet_path': self.source1_paths[idx],
            'mri_path': self.source2_paths[idx],
            'pet_size': Image.open(self.source1_paths[idx]).size,
            'mri_size': Image.open(self.source2_paths[idx]).size
        }
        return info



def create_dataloaders_train(source1_train_img_path,source2_train_img_path,
                             source1_val_img_path,source2_val_img_path,
                             image_size=(256, 256), batch_size=6, num_workers=1
                             ):
    """
    创建训练和验证数据加载器

    Args:
        image_size: 目标图像尺寸
        batch_size: 批次大小
        num_workers: 数据加载进程数

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # 获取图像路径
    source1_train_paths = get_image_paths(source1_train_img_path)
    source2_train_paths = get_image_paths(source2_train_img_path)
    source1_val_paths = get_image_paths(source1_val_img_path)
    source2_val_paths = get_image_paths(source2_val_img_path)

    print(f"训练集: PET={len(source1_train_paths)}, MRI={len(source2_train_paths)}")
    print(f"验证集: PET={len(source1_val_paths)}, MRI={len(source2_val_paths)}")

    # 数据增强配置
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor()
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # 创建数据集
    train_dataset = FusionDataset(
        source1_train_paths,
        source2_train_paths,
        transform=train_transform,
        target_size=image_size
    )

    val_dataset = FusionDataset(
        source1_val_paths,
        source2_val_paths,
        transform=val_transform,
        target_size=image_size
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )

    return train_loader, val_loader

def create_dataloaders_test(source1_test_img_path,source2_test_img_path,
                            image_size=(256, 256), batch_size=6, num_workers=2):
    """
    创建训练数据加载器

    Args:
        image_size: 目标图像尺寸
        batch_size: 批次大小
        num_workers: 数据加载进程数

    Returns:
        test_loader
    """
    # 获取图像路径
    source1_test_paths = get_image_paths(source1_test_img_path)
    source2_test_paths = get_image_paths(source2_test_img_path)

    print(f"训练集: PET={len(source1_test_paths)}, MRI={len(source2_test_paths)}")

    # 数据增强配置
    test_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])

    # 创建数据集
    test_dataset = FusionDataset(
        source1_test_paths,
        source2_test_paths,
        transform=test_transform,
        target_size=image_size
    )

    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0
    )

    return test_loader

# 调试和测试函数
def test_dataloader():
    """测试数据加载器功能"""
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders_train()

    print("\n=== 数据加载器测试 ===")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")

    # 测试第一个批次
    for batch_idx, (pet_batch, mri_batch) in enumerate(train_loader):
        print(f"\n第一个批次形状:")
        print(f"PET: {pet_batch.shape}")  # 应该是 [batch_size, 1, H, W]
        print(f"MRI: {mri_batch.shape}")  # 应该是 [batch_size, 1, H, W]
        print(f"数值范围 - PET: [{pet_batch.min():.3f}, {pet_batch.max():.3f}]")
        print(f"数值范围 - MRI: [{mri_batch.min():.3f}, {mri_batch.max():.3f}]")
        break

    # 测试数据集样本
    sample_pet, sample_mri = train_dataset[0]
    print(f"\n单个样本形状:")
    print(f"PET: {sample_pet.shape}")  # 应该是 [1, H, W]
    print(f"MRI: {sample_mri.shape}")  # 应该是 [1, H, W]

    return train_loader, val_loader


if __name__ == "__main__":
    # 运行测试
    train_loader, val_loader = test_dataloader()