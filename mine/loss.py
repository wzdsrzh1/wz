import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        return 1 - (ssim1 + ssim2) / 2


class GradientLoss(nn.Module):
    """
    梯度损失
    用于保持图像的边缘和细节信息，这在医学图像中至关重要
    """

    def __init__(self):
        super(GradientLoss, self).__init__()

    def gradient(self, img):
        """计算图像梯度"""
        # Sobel算子
        sobel_x = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)

        sobel_x = sobel_x.to(img.device).type_as(img)
        sobel_y = sobel_y.to(img.device).type_as(img)

        if img.shape[1] > 1:
            sobel_x = sobel_x.repeat(img.shape[1], 1, 1, 1)
            sobel_y = sobel_y.repeat(img.shape[1], 1, 1, 1)

        grad_x = F.conv2d(img, sobel_x, padding=1, groups=img.shape[1])
        grad_y = F.conv2d(img, sobel_y, padding=1, groups=img.shape[1])

        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        return gradient_magnitude

    def forward(self, fused_img, source_img1, source_img2):
        """
        确保融合图像保留源图像的最强梯度信息
        """
        grad_fused = self.gradient(fused_img)
        grad_src1 = self.gradient(source_img1)
        grad_src2 = self.gradient(source_img2)

        # 取两个源图像中较大的梯度作为目标
        grad_max = torch.max(grad_src1, grad_src2)

        # 最小化融合图像与最大梯度之间的差异
        loss = F.l1_loss(grad_fused, grad_max)
        return loss


class IntensityLoss(nn.Module):
    """
    强度保持损失
    确保融合图像的整体强度分布接近源图像
    """

    def __init__(self):
        super(IntensityLoss, self).__init__()

    def forward(self, fused_img, source_img1, source_img2):
        """
        保持融合图像的强度在源图像范围内
        """
        # 计算源图像的平均强度
        mean_src1 = torch.mean(source_img1)
        mean_src2 = torch.mean(source_img2)
        mean_fused = torch.mean(fused_img)

        # 融合图像的平均强度应该在两个源图像之间
        target_mean = (mean_src1 + mean_src2) / 2
        intensity_loss = F.l1_loss(mean_fused, target_mean)

        return intensity_loss


class PerceptualLoss(nn.Module):
    """
    感知损失
    使用预训练网络提取特征，保持高层语义信息
    医学图像可使用VGG或专门的医学图像预训练模型
    """

    def __init__(self, feature_layers=[3, 8, 15, 22]):
        super(PerceptualLoss, self).__init__()
        from torchvision import models

        vgg = models.vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential()

        # 提取VGG的特定层
        for i, layer in enumerate(vgg):
            self.feature_extractor.add_module(str(i), layer)
            if i == max(feature_layers):
                break

        # 冻结参数
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.feature_layers = feature_layers

    def forward(self, fused_img, source_img1, source_img2):
        """
        计算融合图像与源图像在特征空间的距离
        """
        # 如果是单通道图像，复制为3通道
        if fused_img.shape[1] == 1:
            fused_img = fused_img.repeat(1, 3, 1, 1)
            source_img1 = source_img1.repeat(1, 3, 1, 1)
            source_img2 = source_img2.repeat(1, 3, 1, 1)

        loss = 0
        x_fused = fused_img
        x_src1 = source_img1
        x_src2 = source_img2

        for i, layer in enumerate(self.feature_extractor):
            x_fused = layer(x_fused)
            x_src1 = layer(x_src1)
            x_src2 = layer(x_src2)

            if i in self.feature_layers:
                # 融合图像应该包含两个源图像的特征
                loss += F.l1_loss(x_fused, x_src1) + F.l1_loss(x_fused, x_src2)

        return loss / len(self.feature_layers)


class MutualInformationLoss(nn.Module):
    """
    互信息损失
    最大化融合图像与源图像之间的互信息
    适用于多模态医学图像融合（如CT和MRI）
    """

    def __init__(self, bins=256, sigma=0.4):
        super(MutualInformationLoss, self).__init__()
        self.bins = bins
        self.sigma = sigma

    def marginal_pdf(self, values):
        """计算边缘概率密度"""
        residuals = values - values.unsqueeze(1)
        kernel_values = torch.exp(-0.5 * (residuals / self.sigma).pow(2))
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf)
        pdf = pdf / (normalization + 1e-10)
        return pdf

    def joint_pdf(self, values1, values2):
        """计算联合概率密度"""
        residuals1 = values1 - values1.unsqueeze(1)
        residuals2 = values2 - values2.unsqueeze(1)

        kernel_values1 = torch.exp(-0.5 * (residuals1 / self.sigma).pow(2))
        kernel_values2 = torch.exp(-0.5 * (residuals2 / self.sigma).pow(2))

        kernel_values = kernel_values1 * kernel_values2
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf)
        pdf = pdf / (normalization + 1e-10)
        return pdf

    def mutual_information(self, img1, img2):
        """计算互信息"""
        img1_flat = img1.view(-1)
        img2_flat = img2.view(-1)

        # 采样以减少计算量
        sample_size = min(10000, img1_flat.shape[0])
        indices = torch.randperm(img1_flat.shape[0])[:sample_size]

        img1_sample = img1_flat[indices]
        img2_sample = img2_flat[indices]

        # 计算概率密度
        pdf_img1 = self.marginal_pdf(img1_sample)
        pdf_img2 = self.marginal_pdf(img2_sample)
        pdf_joint = self.joint_pdf(img1_sample, img2_sample)

        # 计算互信息
        mi = torch.sum(pdf_joint * torch.log((pdf_joint + 1e-10) /
                                             (pdf_img1 * pdf_img2 + 1e-10)))
        return mi

    def forward(self, fused_img, source_img1, source_img2):
        """
        最大化融合图像与源图像的互信息
        返回负互信息作为损失（因为要最小化损失）
        """
        mi1 = self.mutual_information(fused_img, source_img1)
        mi2 = self.mutual_information(fused_img, source_img2)

        # 返回负值以便最小化
        return -(mi1 + mi2)


class MedicalImageFusionLoss(nn.Module):
    """
    综合医学图像融合损失函数
    整合多个损失项，适用于不同类型的医学图像融合任务
    """

    def __init__(self,
                 w_ssim=1.0,
                 w_gradient=10.0,
                 w_intensity=5.0,
                 w_perceptual=1.0,
                 w_mi=0.5,
                 use_perceptual=False):
        """
        Args:
            w_ssim: SSIM损失权重
            w_gradient: 梯度损失权重
            w_intensity: 强度损失权重
            w_perceptual: 感知损失权重
            w_mi: 互信息损失权重
            use_perceptual: 是否使用感知损失（需要预训练模型）
        """
        super(MedicalImageFusionLoss, self).__init__()

        self.w_ssim = w_ssim
        self.w_gradient = w_gradient
        self.w_intensity = w_intensity
        self.w_perceptual = w_perceptual
        self.w_mi = w_mi
        self.use_perceptual = use_perceptual

        self.ssim_loss = StructuralSimilarityLoss()
        self.gradient_loss = GradientLoss()
        self.intensity_loss = IntensityLoss()
        self.mi_loss = MutualInformationLoss()

        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()

    def forward(self, fused_img, source_img1, source_img2):
        """
        计算总损失

        Args:
            fused_img: 融合图像 [B, C, H, W]
            source_img1: 源图像1 [B, C, H, W]
            source_img2: 源图像2 [B, C, H, W]

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        total_loss = 0

        # SSIM损失
        if self.w_ssim > 0:
            ssim_loss = self.ssim_loss(fused_img, source_img1, source_img2)
            loss_dict['ssim'] = ssim_loss.item()
            total_loss += self.w_ssim * ssim_loss

        # 梯度损失
        if self.w_gradient > 0:
            gradient_loss = self.gradient_loss(fused_img, source_img1, source_img2)
            loss_dict['gradient'] = gradient_loss.item()
            total_loss += self.w_gradient * gradient_loss

        # 强度损失
        if self.w_intensity > 0:
            intensity_loss = self.intensity_loss(fused_img, source_img1, source_img2)
            loss_dict['intensity'] = intensity_loss.item()
            total_loss += self.w_intensity * intensity_loss

        # 感知损失
        if self.use_perceptual and self.w_perceptual > 0:
            perceptual_loss = self.perceptual_loss(fused_img, source_img1, source_img2)
            loss_dict['perceptual'] = perceptual_loss.item()
            total_loss += self.w_perceptual * perceptual_loss

        # 互信息损失
        if self.w_mi > 0:
            mi_loss = self.mi_loss(fused_img, source_img1, source_img2)
            loss_dict['mi'] = mi_loss.item()
            total_loss += self.w_mi * mi_loss

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# 使用示例
if __name__ == "__main__":
    # 创建模拟数据
    batch_size = 2
    channels = 1
    height, width = 256, 256

    fused = torch.randn(batch_size, channels, height, width)
    source1 = torch.randn(batch_size, channels, height, width)
    source2 = torch.randn(batch_size, channels, height, width)

    # 单独测试各个损失
    print("=== 测试各个损失函数 ===\n")

    # SSIM损失
    ssim_loss_fn = StructuralSimilarityLoss()
    ssim_loss = ssim_loss_fn(fused, source1, source2)
    print(f"SSIM Loss: {ssim_loss.item():.4f}")

    # 梯度损失
    gradient_loss_fn = GradientLoss()
    gradient_loss = gradient_loss_fn(fused, source1, source2)
    print(f"Gradient Loss: {gradient_loss.item():.4f}")

    # 强度损失
    intensity_loss_fn = IntensityLoss()
    intensity_loss = intensity_loss_fn(fused, source1, source2)
    print(f"Intensity Loss: {intensity_loss.item():.4f}")

    # 互信息损失
    mi_loss_fn = MutualInformationLoss()
    mi_loss = mi_loss_fn(fused, source1, source2)
    print(f"Mutual Information Loss: {mi_loss.item():.4f}")

    # 综合损失
    print("\n=== 测试综合损失函数 ===\n")
    fusion_loss = MedicalImageFusionLoss(
        w_ssim=1.0,
        w_gradient=10.0,
        w_intensity=5.0,
        w_mi=0.5,
        use_perceptual=False  # 设为True需要下载VGG模型
    )

    total_loss, loss_dict = fusion_loss(fused, source1, source2)

    print("各项损失:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    print(f"\n总损失: {total_loss.item():.4f}")

    # 反向传播测试
    total_loss.backward()
    print("\n✓ 反向传播成功!")