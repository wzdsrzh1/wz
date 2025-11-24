import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from config import Config

#存放图像地址的列表
def img_to_list(image_path):
    file_name = os.listdir(image_path)
    length = len(file_name)
    img_path_list = []
    for i in range(length):
        path = os.path.join(image_path, file_name[i])
        img_path_list.append(path)
    return img_path_list

#加载图像
def img_load():
    image_path = Config.pet_train_img_path
    img_path_list = img_to_list(image_path)
    length = len(img_path_list)
    img_list = []
    for i in range(length):
        img = Image.open(img_path_list[i])
        img_list.append(img)
    return img_list

class fusiondataset(Dataset):

    def __init__(self, source1_paths, source2_paths,transform = None, target_size=(256, 256)):
        self.source1_paths = source1_paths
        self.source2_paths = source2_paths
        self.target_size = target_size
        self.transform = transform or transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.source1_paths)

    def __getitem__(self, idx):
        source1 = Image.open(self.source1_paths[idx]).convert('RGB')
        source1 = self.transform(source1)
        source2 = Image.open(self.source2_paths[idx]).convert('RGB')
        source2 = self.transform(source2)

        return source1, source2

def dataloader(image_size = (256, 256),batch_size = 6):
    #两种模态图像的地址
    pet_train_img_paths = img_to_list(Config.pet_train_img_path)
    mri_train_img_paths = img_to_list(Config.mri_train_img_path)
    pet_val_img_paths = img_to_list(Config.pet_val_img_path)
    mri_val_img_paths = img_to_list(Config.mri_val_img_path)
    #数据增强
    train_transform = transforms.Compose([transforms.Resize(image_size),
                                          transforms.RandomHorizontalFlip(0.5),
                                          transforms.RandomRotation(10),
                                          transforms.ToTensor()
                                          ])
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    train_dataset = fusiondataset(pet_train_img_paths, mri_train_img_paths, transform = train_transform,
                                  target_size= image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # 训练时打乱数据
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # 丢弃最后不完整的批次
    )

    val_loader = None
    if pet_val_img_paths and mri_val_img_paths:
        val_dataset = fusiondataset(
            pet_val_img_paths, mri_val_img_paths, transform = val_transform,
            target_size=image_size
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,  # 验证时不打乱
            pin_memory=True if torch.cuda.is_available() else False
        )

    return train_loader, val_loader, train_dataset, val_dataset