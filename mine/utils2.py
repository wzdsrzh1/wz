import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
import cv2
from torchvision import transforms
import SimpleITK as sitk
import nibabel as nib
from skimage import exposure
import pydicom

def normalize_image(image, method='minmax', target_range=(0, 255)):
    """
    使用各种方法对图像进行归一化

    Args:
        image: 输入图像数组
        method: 'minmax':线性归一化到[0,1]，再映射到target_range
                'zscore':按均值或者标准差做Z-score标准化，把结果缩放到target_range
                'clahe':自适应直方图均衡 (CLAHE)，提升对比度,会先把图像归一化到 [0,255]，再用 OpenCV 的 CLAHE
                'percentile':用 2% ~ 98% 分位数做裁剪，减小异常值影响,裁剪后再线性映射到 target_range
        target_range: 归一化数值区间，默认（0,255）

    Returns:
        归一化后的图像
    """
    image = image.astype(np.float32)

    if method == 'minmax':
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val - min_val > 0:
            image = (image - min_val) / (max_val - min_val)
            image = image * (target_range[1] - target_range[0]) + target_range[0]

    elif method == 'zscore':
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
            # Scale to target range
            image = (image - image.min()) / (image.max() - image.min())
            image = image * (target_range[1] - target_range[0]) + target_range[0]

    elif method == 'clahe':
        # Contrast Limited Adaptive Histogram Equalization
        image_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image_uint8).astype(np.float32)

    elif method == 'percentile':
        # Percentile normalization (robust to outliers)
        p2, p98 = np.percentile(image, (2, 98))
        image = np.clip(image, p2, p98)
        image = (image - p2) / (p98 - p2)
        image = image * (target_range[1] - target_range[0]) + target_range[0]

    return image


def histogram_matching(source, template):
    """
    Match histogram of source image to template image
    Useful for normalizing images from different scanners/protocols
    """
    matched = exposure.match_histograms(source, template)
    return matched

def modality_specific_preprocess(image, modality):
    """
    图像预处理

    Args:
        image: 输入图像
        modality: 'PET', 'MRI', 'CT', 'SPECT'
    """
    modality = modality.upper()

    if modality == 'CT':
        # CT: windowing for soft tissue
        # Typical HU window: center=40, width=400
        pass  # windowing already handled in load_medical_image

    elif modality == 'PET':
        # PET: log transformation to enhance low-intensity regions
        image = image - image.min() + 1  # Avoid log(0)
        image = np.log(image + 1)

    elif modality == 'MRI':
        # MRI: bias field correction could be applied here
        # For now, just ensure positive values
        image = image - image.min()

    elif modality == 'SPECT':
        # SPECT: similar to PET
        image = image - image.min() + 1
        image = np.log(image + 1)

    return image


def augment_medical_image(image, flip_prob=0.5, rotate_prob=0.5):
    """
    图像数据增强
    有限增强以保持医学特征
    """
    # 随机水平翻转
    if random.random() < flip_prob:
        image = np.fliplr(image)

    # 随机垂直反转
    if random.random() < flip_prob:
        image = np.flipud(image)

    # 随机旋转 (90, 180, 270)
    if random.random() < rotate_prob:
        k = random.randint(1, 3)
        image = np.rot90(image, k)

    return image

def save_medical_image(path, data, format='png', preserve_range=True):
    """
    Save medical image in various formats

    Args:
        path: save path
        data: image data (numpy array)
        format: 'png', 'nifti', 'dicom'
        preserve_range: preserve original intensity range
    """
    # Ensure 2D
    if len(data.shape) == 3 and data.shape[2] == 1:
        data = data.reshape([data.shape[0], data.shape[1]])

    if format == 'png':
        if preserve_range:
            # Normalize to 0-255 for display
            data_norm = ((data - data.min()) / (data.max() - data.min() + 1e-8) * 255).astype(np.uint8)
        else:
            data_norm = data.astype(np.uint8)

        cv2.imwrite(path, data_norm)

    elif format == 'nifti':
        # Save as NIfTI
        nii_img = nib.Nifti1Image(data, np.eye(4))
        nib.save(nii_img, path)

    elif format == 'dicom':
        # Basic DICOM save (requires more metadata for real use)
        print("Warning: DICOM save requires proper metadata. Saving as PNG instead.")
        save_medical_image(path.replace('.dcm', '.png'), data, format='png')


def save_fusion_results(save_dir, image_data, modality1_data, modality2_data,
                        filename, save_comparison=True):
    """
    Save fusion results with optional comparison visualization

    Args:
        save_dir: directory to save results
        image_data: fused image
        modality1_data: source image 1
        modality2_data: source image 2
        filename: base filename
        save_comparison: whether to save side-by-side comparison
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save fused image
    fused_path = join(save_dir, f'{filename}_fused.png')
    save_medical_image(fused_path, image_data)

    if save_comparison:
        # Create comparison image
        h, w = image_data.shape[:2]
        comparison = np.zeros((h, w * 3), dtype=np.uint8)

        # Normalize all to 0-255
        mod1_norm = ((modality1_data - modality1_data.min()) /
                     (modality1_data.max() - modality1_data.min() + 1e-8) * 255).astype(np.uint8)
        mod2_norm = ((modality2_data - modality2_data.min()) /
                     (modality2_data.max() - modality2_data.min() + 1e-8) * 255).astype(np.uint8)
        fused_norm = ((image_data - image_data.min()) /
                      (image_data.max() - image_data.min() + 1e-8) * 255).astype(np.uint8)

        if len(mod1_norm.shape) == 3:
            mod1_norm = mod1_norm.reshape(h, w)
        if len(mod2_norm.shape) == 3:
            mod2_norm = mod2_norm.reshape(h, w)
        if len(fused_norm.shape) == 3:
            fused_norm = fused_norm.reshape(h, w)

        comparison[:, :w] = mod1_norm
        comparison[:, w:2 * w] = mod2_norm
        comparison[:, 2 * w:3 * w] = fused_norm

        # Add labels
        comparison_color = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
        cv2.putText(comparison_color, 'Modality 1', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison_color, 'Modality 2', (w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison_color, 'Fused', (2 * w + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        comparison_path = join(save_dir, f'{filename}_comparison.png')
        cv2.imwrite(comparison_path, comparison_color)

# Compatibility functions for tensor operations
def tensor_save_image(tensor, filename, cuda=True):
    """
    将图像保存为张量
    """
    if cuda:
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        img = tensor.clamp(0, 255).data[0].numpy()

    if len(img.shape) == 3:
        if img.shape[0] == 1:
            img = img[0]
        else:
            img = img.transpose(1, 2, 0)

    img = img.astype(np.uint8)
    cv2.imwrite(filename, img)


def gram_matrix(y):
    """
    Compute Gram matrix for style transfer (if needed)
    """
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram