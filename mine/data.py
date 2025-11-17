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


def list_images(directory, extensions=None):
    """
    返回目录中图像的名称和后缀（List all medical image files in directory）
    Supports: PNG, JPG, JPEG, NIfTI (.nii, .nii.gz), TIFF
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
                # 提取不带后缀的图像名称（Extract filename without extension）
                if ext == '.nii.gz':
                    name_clean = name.replace('.nii.gz', '')
                else:
                    name_clean = splitext(name)[0]
                names.append(name_clean)
                break

    return images, names


def load_medical_image(filepath, normalize=True, target_size=None):
    """
    加载各种格式的医学图像（Load medical images from various formats） (DICOM, NIfTI, PNG, etc.)

    Args:
        filepath: 图像文件路径（path to the image file）
        normalize: 归一化尺寸（whether to normalize to [0, 1] or [0, 255]）
        target_size: 图像重设尺寸 tuple (H, W) for resizing, None to keep original

    Returns:
        图像的numpy数组 （numpy array of the image）
    """
    ext = filepath.lower()

    # NIfTI format
    if ext.endswith('.nii') or ext.endswith('.nii.gz'):
        nii_img = nib.load(filepath)
        image = nii_img.get_fdata().astype(np.float32)

        #如果是3D体积，提取中间切片或者平均 If 3D volume, extract middle slice or average
        if len(image.shape) == 3:
            if image.shape[2] > 1:
                image = image[:, :, image.shape[2] // 2]
            else:
                image = np.squeeze(image)

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
        image = normalize_image(image)

    return image


def normalize_image(image, method='minmax', target_range=(0, 255)):
    """
    Normalize medical images with various methods

    Args:
        image: input image array
        method: 'minmax', 'zscore', 'clahe', 'percentile'
        target_range: target intensity range

    Returns:
        normalized image
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


def get_train_images_medical(paths, height=256, width=256, normalize_method='minmax',
                             augment=False, modality=None):
    """
    Load training images for medical fusion task

    Args:
        paths: list of image paths
        height, width: target size
        normalize_method: normalization method
        augment: whether to apply data augmentation
        modality: 'PET', 'MRI', 'CT', 'SPECT' for modality-specific preprocessing

    Returns:
        torch tensor of images
    """
    if isinstance(paths, str):
        paths = [paths]

    images = []
    for path in paths:
        # Load medical image
        image = load_medical_image(path, normalize=False, target_size=(width, height))

        # Modality-specific preprocessing
        if modality:
            image = modality_specific_preprocess(image, modality)

        # Normalize
        image = normalize_image(image, method=normalize_method)

        # Data augmentation for training
        if augment:
            image = augment_medical_image(image)

        # Add channel dimension
        image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_test_images_medical(paths, height=None, width=None, normalize_method='minmax', modality=None):
    """
    Load test images for medical fusion task
    Maintains original size if height/width not specified
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
            image = modality_specific_preprocess(image, modality)

        # Normalize
        image = normalize_image(image, method=normalize_method)

        # Add channel dimension
        image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def modality_specific_preprocess(image, modality):
    """
    Apply modality-specific preprocessing

    Args:
        image: input image
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
    Data augmentation for medical images
    Limited augmentation to preserve medical characteristics
    """
    # Random horizontal flip
    if random.random() < flip_prob:
        image = np.fliplr(image)

    # Random vertical flip
    if random.random() < flip_prob:
        image = np.flipud(image)

    # Random rotation (90, 180, 270 degrees only)
    if random.random() < rotate_prob:
        k = random.randint(1, 3)
        image = np.rot90(image, k)

    return image


def load_dataset_medical(image_path_modality1, image_path_modality2, BATCH_SIZE,
                         num_imgs=None, shuffle=True):
    """
    Load paired medical images for fusion training

    Args:
        image_path_modality1: list of paths for modality 1 (e.g., PET)
        image_path_modality2: list of paths for modality 2 (e.g., MRI)
        BATCH_SIZE: batch size
        num_imgs: number of images to use (None for all)
        shuffle: whether to shuffle the data

    Returns:
        paired image paths and number of batches
    """
    assert len(image_path_modality1) == len(image_path_modality2), \
        "Modality 1 and Modality 2 must have same number of images"

    if num_imgs is None:
        num_imgs = len(image_path_modality1)

    # Pair the images
    paired_paths = list(zip(image_path_modality1[:num_imgs],
                            image_path_modality2[:num_imgs]))

    # Shuffle
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


def calculate_metrics(fused, source1, source2):
    """
    Calculate common fusion quality metrics

    Returns:
        dict with metrics: EN (Entropy), MI (Mutual Information),
        SF (Spatial Frequency), SSIM
    """
    from skimage.metrics import structural_similarity as ssim

    metrics = {}

    # Entropy
    hist, _ = np.histogram(fused.flatten(), bins=256, range=(0, 255))
    hist = hist / hist.sum()
    metrics['EN'] = -np.sum(hist * np.log2(hist + 1e-7))

    # Mutual Information (simplified)
    hist_joint, _, _ = np.histogram2d(source1.flatten(), fused.flatten(), bins=256)
    pxy = hist_joint / hist_joint.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    metrics['MI'] = np.sum(pxy[nzs] * np.log2(pxy[nzs] / (px_py[nzs] + 1e-7)))

    # Spatial Frequency
    rf = np.diff(fused, axis=0)
    cf = np.diff(fused, axis=1)
    metrics['SF'] = np.sqrt(np.mean(rf ** 2) + np.mean(cf ** 2))

    # SSIM with both sources
    metrics['SSIM1'] = ssim(source1, fused, data_range=255)
    metrics['SSIM2'] = ssim(source2, fused, data_range=255)

    return metrics


# Compatibility functions for tensor operations
def tensor_save_image(tensor, filename, cuda=True):
    """Save tensor as image file"""
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
    """Compute Gram matrix for style transfer (if needed)"""
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram