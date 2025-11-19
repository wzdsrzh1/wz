import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from args_fusion import args

#存放图像地址的列表
def img_to_list(image_path):
    file_name = os.listdir(image_path)
    length = len(file_name)
    img_path_list = []
    for i in range(length):
        path = os.path.join(image_path, file_name[i - 1])
        img_path_list.append(path)
    return img_path_list

#把图像转换为张量
def img_to_tensor(img,target_size = None):
    transform_list = []
    if target_size:
        transform_list.append(transforms.Resize(target_size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    tensor_img = transform(img)
    if len(tensor_img.size()) == 3:
        tensor_img = tensor_img.unsqueeze(0)
    return tensor_img

#加载图像
def img_load(image_path):
    img_path_list = img_to_list(image_path)
    length = len(img_path_list)
    img_list = []
    for i in range(length):
        img = Image.open(img_path_list[i - 1])
        img_list.append(img)
    return img_list

class dataset():

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
        if len(source1.shape) == 3:
            source1 = source1.unsqueeze(0)
        source2 = Image.open(self.source2_paths[idx]).convert('RGB')
        source2 = self.transform(source2)
        if len(source2.shape) == 3:
            source2 = source2.unsqueeze(0)

        return source1, source2

