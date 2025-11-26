# 改进的互信息损失函数 - 解决训练/验证差异问题
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MutualInformationLossFixed(nn.Module):
    """
    改进的互信息损失 - 解决训练/验证集差异大的问题
    
    主要改进：
    1. 固定随机种子或使用全部像素点
    2. 自适应sigma，根据数据范围调整
    3. 改进数值稳定性
    4. 可选的确定性模式（验证时使用）
    """

    def __init__(self, bins=256, sigma=None, use_adaptive_sigma=True, 
                 use_all_pixels=False, deterministic=False, sample_size=10000):
        """
        Args:
            bins: 直方图bins数（如果使用直方图方法）
            sigma: 高斯核参数，如果None则自适应
            use_adaptive_sigma: 是否使用自适应sigma
            use_all_pixels: 是否使用全部像素点（True更稳定但慢）
            deterministic: 是否使用确定性模式（固定随机种子）
            sample_size: 采样大小（如果use_all_pixels=False）
        """
        super(MutualInformationLossFixed, self).__init__()
        self.bins = bins
        self.sigma = sigma
        self.use_adaptive_sigma = use_adaptive_sigma
        self.use_all_pixels = use_all_pixels
        self.deterministic = deterministic
        self.sample_size = sample_size

    def get_adaptive_sigma(self, values):
        """
        根据数据范围自适应计算sigma
        使用数据标准差的固定比例
        """
        std = torch.std(values)
        mean = torch.mean(values)
        
        # 方法1：使用标准差的比例
        if self.sigma is None:
            # 默认使用标准差的20%作为sigma
            adaptive_sigma = std * 0.2
        else:
            # 如果提供了sigma，根据数据范围调整
            data_range = torch.max(values) - torch.min(values)
            adaptive_sigma = self.sigma * (data_range / 1.0)  # 假设归一化到[0,1]
        
        # 防止sigma过小导致数值问题
        adaptive_sigma = torch.clamp(adaptive_sigma, min=1e-3, max=1.0)
        
        return adaptive_sigma

    def marginal_pdf_stable(self, values, sigma):
        """
        改进的边缘概率密度计算 - 数值稳定版本
        """
        # 归一化数据到合理范围
        values_norm = (values - torch.mean(values)) / (torch.std(values) + 1e-8)
        
        # 计算残差矩阵
        residuals = values_norm - values_norm.unsqueeze(1)  # [N, N]
        
        # 计算高斯核
        kernel_values = torch.exp(-0.5 * (residuals / sigma).pow(2))
        
        # 对每一行求平均（核密度估计）
        pdf = torch.mean(kernel_values, dim=1)  # [N]
        
        # 归一化
        normalization = torch.sum(pdf)
        pdf = pdf / (normalization + 1e-10)
        
        return pdf

    def joint_pdf_stable(self, values1, values2, sigma1, sigma2):
        """
        改进的联合概率密度计算 - 数值稳定版本
        """
        # 归一化数据
        values1_norm = (values1 - torch.mean(values1)) / (torch.std(values1) + 1e-8)
        values2_norm = (values2 - torch.mean(values2)) / (torch.std(values2) + 1e-8)
        
        # 计算残差矩阵
        residuals1 = values1_norm - values1_norm.unsqueeze(1)  # [N, N]
        residuals2 = values2_norm - values2_norm.unsqueeze(1)  # [N, N]
        
        # 计算高斯核
        kernel_values1 = torch.exp(-0.5 * (residuals1 / sigma1).pow(2))
        kernel_values2 = torch.exp(-0.5 * (residuals2 / sigma2).pow(2))
        
        kernel_values = kernel_values1 * kernel_values2
        
        # 对每一行求平均
        pdf = torch.mean(kernel_values, dim=1)  # [N]
        
        # 归一化
        normalization = torch.sum(pdf)
        pdf = pdf / (normalization + 1e-10)
        
        return pdf

    def mutual_information_histogram(self, img1, img2):
        """
        基于直方图的互信息计算（更稳定但精度略低）
        """
        # 展平图像
        img1_flat = img1.view(-1)
        img2_flat = img2.view(-1)
        
        # 归一化到[0, bins-1]
        img1_min, img1_max = img1_flat.min(), img1_flat.max()
        img2_min, img2_max = img2_flat.min(), img2_flat.max()
        
        img1_norm = (img1_flat - img1_min) / (img1_max - img1_min + 1e-8)
        img2_norm = (img2_flat - img2_min) / (img2_max - img2_min + 1e-8)
        
        img1_binned = (img1_norm * (self.bins - 1)).long().clamp(0, self.bins - 1)
        img2_binned = (img2_norm * (self.bins - 1)).long().clamp(0, self.bins - 1)
        
        # 计算联合直方图
        joint_hist = torch.zeros(self.bins, self.bins, device=img1.device)
        joint_hist.index_put_(
            (img1_binned, img2_binned),
            torch.ones_like(img1_binned, dtype=torch.float32),
            accumulate=True
        )
        joint_hist = joint_hist / (joint_hist.sum() + 1e-10)
        
        # 计算边缘直方图
        marginal1 = joint_hist.sum(dim=1)
        marginal2 = joint_hist.sum(dim=0)
        
        # 计算互信息
        nz = joint_hist > 0
        mi = torch.sum(
            joint_hist[nz] * torch.log(
                (joint_hist[nz] + 1e-10) / 
                ((marginal1[img1_binned[nz]] * marginal2[img2_binned[nz]]) + 1e-10)
            )
        )
        
        return mi

    def mutual_information_kde(self, img1, img2):
        """
        基于核密度估计的互信息计算（原方法改进版）
        """
        img1_flat = img1.view(-1)
        img2_flat = img2.view(-1)
        
        # 采样策略
        total_pixels = img1_flat.shape[0]
        
        if self.use_all_pixels:
            # 使用全部像素点
            img1_sample = img1_flat
            img2_sample = img2_flat
        else:
            # 随机采样
            sample_size = min(self.sample_size, total_pixels)
            
            if self.deterministic:
                # 确定性模式：使用固定索引（前N个像素）
                indices = torch.arange(sample_size, device=img1.device)
            else:
                # 随机模式
                indices = torch.randperm(total_pixels, device=img1.device)[:sample_size]
            
            img1_sample = img1_flat[indices]
            img2_sample = img2_flat[indices]
        
        # 自适应sigma
        if self.use_adaptive_sigma:
            sigma1 = self.get_adaptive_sigma(img1_sample)
            sigma2 = self.get_adaptive_sigma(img2_sample)
        else:
            sigma1 = self.sigma if self.sigma is not None else 0.4
            sigma2 = sigma1
        
        # 计算概率密度
        pdf_img1 = self.marginal_pdf_stable(img1_sample, sigma1)
        pdf_img2 = self.marginal_pdf_stable(img2_sample, sigma2)
        pdf_joint = self.joint_pdf_stable(img1_sample, img2_sample, sigma1, sigma2)
        
        # 计算互信息
        # 使用更稳定的log计算
        ratio = (pdf_joint + 1e-10) / (pdf_img1 * pdf_img2 + 1e-10)
        log_ratio = torch.log(torch.clamp(ratio, min=1e-10, max=1e10))
        mi = torch.sum(pdf_joint * log_ratio)
        
        return mi

    def forward(self, fused_img, source_img1, source_img2):
        """
        计算互信息损失
        
        Args:
            fused_img: 融合图像 [B, C, H, W]
            source_img1: 源图像1 [B, C, H, W]
            source_img2: 源图像2 [B, C, H, W]
            
        Returns:
            负互信息（作为损失）
        """
        batch_size = fused_img.shape[0]
        total_mi = 0.0
        
        for b in range(batch_size):
            # 计算融合图像与两个源图像的互信息
            mi1 = self.mutual_information_kde(fused_img[b], source_img1[b])
            mi2 = self.mutual_information_kde(fused_img[b], source_img2[b])
            total_mi += (mi1 + mi2)
        
        # 平均
        avg_mi = total_mi / batch_size
        
        # 返回负值以便最小化
        return -avg_mi


# 使用示例和对比测试
if __name__ == "__main__":
    print("测试改进的互信息损失函数...\n")
    
    # 创建测试数据
    batch_size = 2
    channels = 1
    height, width = 256, 256
    
    fused = torch.randn(batch_size, channels, height, width)
    source1 = torch.randn(batch_size, channels, height, width)
    source2 = torch.randn(batch_size, channels, height, width)
    
    # 测试不同配置
    configs = [
        ("原始方法（随机采样）", False, False, False),
        ("确定性模式（固定采样）", False, False, True),
        ("使用全部像素", True, False, False),
        ("自适应sigma", False, True, False),
        ("全部优化", True, True, True),
    ]
    
    print("不同配置下的MI损失值：")
    print("-" * 60)
    
    for name, use_all, adaptive_sigma, deterministic in configs:
        mi_loss_fn = MutualInformationLossFixed(
            use_all_pixels=use_all,
            use_adaptive_sigma=adaptive_sigma,
            deterministic=deterministic
        )
        
        # 多次计算看稳定性
        losses = []
        for _ in range(5):
            loss = mi_loss_fn(fused, source1, source2)
            losses.append(loss.item())
        
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        
        print(f"{name:30s}: {mean_loss:8.4f} ± {std_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("建议：")
    print("1. 训练时使用：use_all_pixels=False, deterministic=False")
    print("2. 验证时使用：use_all_pixels=False, deterministic=True (或use_all_pixels=True)")
    print("3. 始终使用：use_adaptive_sigma=True")
    print("=" * 60)

