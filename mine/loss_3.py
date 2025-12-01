import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=1):
    L = val_range
    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)  # 确保window在正确设备上

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def ssim_loss(fused_img, source1, source2):
    loss_1 = 1 - ssim(fused_img, source1)
    loss_2 = 1 - ssim(fused_img, source2)
    return (loss_1 + loss_2) / 2


def intensity_loss(fused_img, source1, source2):
    max_reference = torch.max(source1, source2)
    return F.l1_loss(fused_img, max_reference)


def gradient_loss(fused_img, source1, source2):
    def image_gradient(img):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)

        if img.size(1) == 3:
            grad_x = torch.cat([F.conv2d(img[:, i:i + 1], sobel_x, padding=1) for i in range(3)], dim=1)
            grad_y = torch.cat([F.conv2d(img[:, i:i + 1], sobel_y, padding=1) for i in range(3)], dim=1)
        else:
            grad_x = F.conv2d(img, sobel_x, padding=1)
            grad_y = F.conv2d(img, sobel_y, padding=1)

        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    grad_fused = image_gradient(fused_img)
    grad_source1 = image_gradient(source1)
    grad_source2 = image_gradient(source2)

    max_grad = torch.max(grad_source1, grad_source2)
    return F.l1_loss(grad_fused, max_grad)


class VGGPerceptualLoss(nn.Module):
    """
    基于VGG的特征相关性损失 (ℒ_deco)
    使用预训练的VGG网络提取特征，计算特征空间的相关性
    """

    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        ))
        blocks.append(nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        ))
        blocks.append(nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        ))

        for block in blocks:
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.constant_(layer.bias, 0)

        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize

        # 注册缓冲区，确保它们与模型在同一设备上
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2]):
        # 如果输入是单通道，扩展到3通道
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        # 归一化 - 确保mean和std与输入在同一设备上
        # 使用注册的缓冲区，它们会自动移动到正确设备
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        loss = 0.0
        x, y = input, target

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)

        return loss


class ImageFusionLoss(nn.Module):
    """
    完整的图像融合损失函数
    ℒ_Fu = ℒ_SSIM + ℒ_deco + ℒ_grad + ℒ_int
    """

    def __init__(self, w_ssim=1.0, w_deco=1.0, w_grad=1.0, w_int=1.0):
        super(ImageFusionLoss, self).__init__()
        self.w_ssim = w_ssim
        self.w_deco = w_deco
        self.w_grad = w_grad
        self.w_int = w_int

        self.ssim_loss = ssim_loss
        self.gradient_loss = gradient_loss
        self.intensity_loss = intensity_loss

        # 创建VGG损失实例，它会自动与模型在同一设备上
        self.vgg_loss = VGGPerceptualLoss()

    def deco_loss(self, fused_img, source1, source2):
        """
        特征相关性损失 (ℒ_deco)
        计算融合图像与两个源图像在特征空间的相关性
        """
        # 计算融合图像与两个源图像的特征相关性
        loss1 = self.vgg_loss(fused_img, source1)
        loss2 = self.vgg_loss(fused_img, source2)

        return (loss1 + loss2) / 2

    def forward(self, fused_img, source1, source2):
        """
        Args:
            fused_img: 融合后的图像 [B, C, H, W]
            source1: 源图像1 (如红外图像) [B, C, H, W]
            source2: 源图像2 (如可见光图像) [B, C, H, W]
        """
        loss_dict = {}
        total_loss = 0

        # 1. 结构相似性损失
        ssim_loss_val = self.ssim_loss(fused_img, source1, source2)
        loss_dict['ssim'] = ssim_loss_val.item()

        # 2. 特征相关性损失
        deco_loss_val = self.deco_loss(fused_img, source1, source2)
        loss_dict['deco'] = deco_loss_val.item()

        # 3. 梯度损失
        grad_loss_val = self.gradient_loss(fused_img, source1, source2)
        loss_dict['grad'] = grad_loss_val.item()

        # 4. 像素强度损失
        int_loss_val = self.intensity_loss(fused_img, source1, source2)
        loss_dict['int'] = int_loss_val.item()

        # 总损失
        total_loss = (self.w_ssim * ssim_loss_val +
                      self.w_deco * deco_loss_val +
                      self.w_grad * grad_loss_val +
                      self.w_int * int_loss_val)

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# 使用示例
if __name__ == "__main__":
    # 模拟输入数据
    batch_size, channels, height, width = 2, 1, 256, 256

    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 生成模拟图像并移动到设备
    fused_img = torch.rand(batch_size, channels, height, width).to(device)
    source1 = torch.rand(batch_size, channels, height, width).to(device)  # 红外图像
    source2 = torch.rand(batch_size, channels, height, width).to(device)  # 可见光图像

    # 初始化损失函数并移动到设备
    loss_fn = ImageFusionLoss(
        w_ssim=1.0,
        w_deco=1.0,
        w_grad=1.0,
        w_int=1.0
    ).to(device)

    # 计算损失
    total_loss, loss_dict = loss_fn(fused_img, source1, source2)

    # 打印结果
    print("图像融合损失分量:")
    for name, value in loss_dict.items():
        print(f"  {name}: {value:.6f}")