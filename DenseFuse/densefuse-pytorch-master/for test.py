import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

class StructuralSimilarityLoss(nn.Module):
    """
    结构相似性损失 (SSIM Loss)
    用于保持融合图像与源图像之间的结构相似性
    """

    def __init__(self, window_size=11, size_average=True):
        super(StructuralSimilarityLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma=1.5):
        """创建高斯窗口"""
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        """创建2D高斯窗口"""
        _1D_window = self.gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def ssim(self, img1, img2):
        """计算SSIM"""
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            window = window.to(img1.device).type_as(img1)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, fused_img, source_img1, source_img2):
        """
        计算融合图像与两个源图像的SSIM损失
        目标: 最大化SSIM，所以返回 1 - SSIM
        """
        ssim1 = self.ssim(fused_img, source_img1)
        ssim2 = self.ssim(fused_img, source_img2)
        print(ssim1, ssim2)
        return 1 - (ssim1 + ssim2) / 2


def load_and_preprocess_image(image_path, target_size=None):
    """
    加载图像并转换为适合SSIM损失的张量格式
    """
    # 打开图像
    pil_img = Image.open(image_path)
    if pil_img.mode == 'L':
        pil_img = pil_img.convert('RGB')

    # 定义转换
    transform_list = []
    if target_size:
        transform_list.append(transforms.Resize(target_size))
    transform_list.append(transforms.ToTensor())

    transform = transforms.Compose(transform_list)

    # 应用转换
    tensor_img = transform(pil_img)

    # 确保是4维 [1, C, H, W]
    if len(tensor_img.size()) == 3:
        tensor_img = tensor_img.unsqueeze(0)

    return tensor_img


fused_img = load_and_preprocess_image('D:/image fusion/swinfusion/SwinFusion-master/results/SwinFusion_PET-MRI/25015.png')
print(fused_img.size())
source_img1 = load_and_preprocess_image('D:/image fusion/swinfusion/SwinFusion-master/Dataset/testsets/PET-MRI/MRI/25015.png')
print(source_img1.size())
source_img2 = load_and_preprocess_image('D:/image fusion/swinfusion/SwinFusion-master/Dataset/testsets/PET-MRI/PET/25015.png')
print(source_img2.size())

loss_ssim = StructuralSimilarityLoss()
sorce = loss_ssim(fused_img, source_img1, source_img2)
print(sorce)