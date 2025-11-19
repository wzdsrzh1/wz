import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
from h5py import Dataset
from torch.autograd import Variable
import cv2
from torchvision import transforms
import SimpleITK as sitk
import nibabel as nib
from skimage import exposure
import pydicom
import utils2

def list_images(directory, extensions=None):
    """
    返回目录中图像的名称和后缀
    支持格式: PNG, JPG, JPEG, NIfTI (.nii, .nii.gz), TIFF
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.nii', '.nii.gz', '.tif', '.tiff']

    images = []
    names = []
    dir_list = listdir(directory)
    dir_list.sort()

    for file in dir_list:
        name = file.lower()
        for ext in extensions:
            if name.endswith(ext):
                images.append(join(directory, file))
                # 提取不带后缀的图像名称
                if ext == '.nii.gz':
                    name_clean = name.replace('.nii.gz', '')
                else:
                    name_clean = splitext(name)[0]
                names.append(name_clean)
                break

    return images, names

def load_medical_image(filepath, normalize=True, target_size=None):
    """
    加载各种格式的医学图像 (DICOM, NIfTI, PNG, etc.)

    Args:
        filepath: 图像文件路径
        normalize: 归一化尺寸（ [0, 1] 或 [0, 255]）
        target_size: 图像重设尺寸 （H,W），target_size = None 时不重设尺寸

    Returns:
        图像的numpy数组 （numpy array of the image）
    """
    ext = filepath.lower()

    # NIfTI format
    if ext.endswith('.nii') or ext.endswith('.nii.gz'):
        nii_img = nib.load(filepath)
        image = nii_img.get_fdata().astype(np.float32)

    # Standard image formats
    elif ext.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            image = np.array(Image.open(filepath).convert('RGB'))
        image = image.astype(np.float32)

    else:
        raise ValueError(f"Unsupported file format: {filepath}")

    # Resize if needed
    if target_size is not None:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Normalize
    if normalize:
        image = utils2.normalize_image(image)

    return image

def get_train_images_medical(paths, height=256, width=256, normalize_method='minmax',
                             augment=False, modality=None):
    """
    加载训练数据集

    Args:
        paths: 图像地址列表
        height, width: 图像高度和宽度
        normalize_method: 归一化方法
        augment: 是否应用数据增强
        modality: 'PET', 'MRI', 'CT', 'SPECT' 的特定模态预处理方法

    Returns:
        图像张量
    """
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        # Load medical image
        image = load_medical_image(path, normalize=False, target_size=(width, height))

        # Modality-specific preprocessing
        if modality:
            image = utils2.modality_specific_preprocess(image, modality)

        # Normalize
        image = utils2.normalize_image(image, method=normalize_method)

        # Data augmentation for training
        if augment:
            image = utils2.augment_medical_image(image)

        # Add channel dimension
        image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_test_images_medical(paths, height=None, width=None, normalize_method='minmax', modality=None):
    """
    加载测试数据集
    如果没有指定大小则保持原尺寸
    """
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        # Load medical image
        if height is not None and width is not None:
            target_size = (width, height)
        else:
            target_size = None

        image = load_medical_image(path, normalize=False, target_size=target_size)

        # Modality-specific preprocessing
        if modality:
            image = utils2.modality_specific_preprocess(image, modality)

        # Normalize
        image = utils2.normalize_image(image, method=normalize_method)

        # Add channel dimension
        image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def load_dataset_medical(image_path_modality1, image_path_modality2, BATCH_SIZE,
                         num_imgs=None, shuffle=True):
    """
    加载融合训练的图像对

    Args:
        image_path_modality1: 模态1的图像路径列表 (e.g., PET)
        image_path_modality2: 模态2的图像路径列表 (e.g., MRI)
        BATCH_SIZE: 批次大小
        num_imgs: 使用图像的数量，num_imgs = None时使用所有图像
        shuffle: 是否打乱数据

    Returns:
        图像对路径和批量大小
    """
    assert len(image_path_modality1) == len(image_path_modality2), \
        "两个模态的图像数量必须相同"

    if num_imgs is None:
        num_imgs = len(image_path_modality1)

    # Pair the images
    paired_paths = list(zip(image_path_modality1[:num_imgs],
                            image_path_modality2[:num_imgs]))

    # 打乱数据
    if shuffle:
        random.shuffle(paired_paths)

    mod = num_imgs % BATCH_SIZE
    print(f'BATCH SIZE: {BATCH_SIZE}')
    print(f'Train images number: {num_imgs}')
    print(f'Train batches: {num_imgs // BATCH_SIZE}')

    if mod > 0:
        print(f'Train set trimmed {mod} samples to fit batch size\n')
        paired_paths = paired_paths[:-mod]

    batches = len(paired_paths) // BATCH_SIZE
    return paired_paths, batches

class FusionDataset(Dataset):

    def __init__(self, image_path_modality1, image_path_modality2, batch_size):
        super().__init__()
        self.image_path_modality1 = image_path_modality1
        self.image_path_modality2 = image_path_modality2
        self.batch_size = batch_size