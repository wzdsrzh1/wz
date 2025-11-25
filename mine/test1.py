# 医学图像融合测试脚本
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import csv
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MedicalFusion_net
from loss import MedicalImageFusionLoss
from dataload_1 import create_dataloaders_test, get_image_paths
from args_fusion import args
from metrics import compute_metrics, entropy, spatial_frequency, std_deviation, mutual_information


class Tester:
    """医学图像融合模型测试器"""
    
    def __init__(self, config):
        """
        初始化测试器
        
        Args:
            config: 配置类对象（args_fusion.args）
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'fused_images'), exist_ok=True)
        if config.save_comparison:
            os.makedirs(os.path.join(config.output_dir, 'comparisons'), exist_ok=True)
        
        # 加载模型
        self.model = self.load_model(config.model_path if config.checkpoint_path is None else config.checkpoint_path)
        
        # 初始化损失函数（用于测试时计算损失）
        self.criterion = MedicalImageFusionLoss(
            w_ssim=config.w_ssim,
            w_gradient=config.w_gradient,
            w_intensity=config.w_intensity,
            w_perceptual=config.w_perceptual,
            w_mi=config.w_mi,
            use_perceptual=config.use_perceptual
        ).to(self.device)
        
        print(f"\n{'='*60}")
        print(f"测试器初始化完成")
        print(f"设备: {self.device}")
        print(f"融合策略: {config.fusion_strategy}")
        print(f"输出目录: {config.output_dir}")
        print(f"{'='*60}\n")
    
    def load_model(self, model_path):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载好的模型
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        print(f"加载模型: {model_path}")
        
        # 创建模型
        model = MedicalFusion_net(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc
        ).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 检查是否是检查点格式（包含state_dict）还是直接是state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"从检查点加载模型 (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("加载模型权重")
        
        model.eval()
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,} ({total_params * 4 / 1e6:.2f}MB)\n")
        
        return model
    
    def tensor_to_numpy(self, tensor):
        """
        将tensor转换为numpy数组
        
        Args:
            tensor: PyTorch tensor [C, H, W] 或 [B, C, H, W]
            
        Returns:
            numpy数组 [H, W] 或 [H, W, C]
        """
        if tensor.dim() == 4:
            tensor = tensor[0]  # 取第一个batch
        
        # 转换到CPU
        img_np = tensor.detach().cpu().numpy()
        
        if tensor.dim() == 3:
            # [C, H, W] -> [H, W, C]
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(2)  # [H, W, 1] -> [H, W]
        
        # ToTensor()会将数据归一化到[0,1]，需要转换回0-255
        # 但如果数据已经是0-255范围，则不需要转换
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        return img_np
    
    def save_image(self, img_np, save_path):
        """
        保存图像
        
        Args:
            img_np: numpy数组 [H, W] 或 [H, W, C]
            save_path: 保存路径
        """
        if img_np.ndim == 2:
            # 灰度图
            img = Image.fromarray(img_np, mode='L')
        elif img_np.ndim == 3:
            if img_np.shape[2] == 1:
                img = Image.fromarray(img_np.squeeze(2), mode='L')
            elif img_np.shape[2] == 3:
                img = Image.fromarray(img_np, mode='RGB')
            else:
                raise ValueError(f"不支持的通道数: {img_np.shape[2]}")
        else:
            raise ValueError(f"不支持的图像维度: {img_np.ndim}")
        
        img.save(save_path)
    
    def save_comparison(self, img1_np, img2_np, fused_np, save_path, img_idx):
        """
        保存对比图（源图像1、源图像2、融合图像）
        
        Args:
            img1_np: 源图像1 (numpy数组)
            img2_np: 源图像2 (numpy数组)
            fused_np: 融合图像 (numpy数组)
            save_path: 保存路径
            img_idx: 图像索引
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 转换为灰度显示
        def to_gray(img):
            if img.ndim == 3:
                return np.mean(img, axis=2)
            return img
        
        axes[0].imshow(to_gray(img1_np), cmap='gray')
        axes[0].set_title('Source 1')
        axes[0].axis('off')
        
        axes[1].imshow(to_gray(img2_np), cmap='gray')
        axes[1].set_title('Source 2')
        axes[1].axis('off')
        
        axes[2].imshow(to_gray(fused_np), cmap='gray')
        axes[2].set_title('Fused')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def test_single_batch(self, img1, img2, img_idx=None):
        """
        测试单个批次
        
        Args:
            img1: 源图像1 [B, C, H, W]
            img2: 源图像2 [B, C, H, W]
            img_idx: 图像索引（可选）
            
        Returns:
            fused_img: 融合图像 [B, C, H, W]
            loss_dict: 损失字典
        """
        # 移动到设备
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)
        
        # 通道数转换（确保与模型输入匹配）
        # dataload_1.py返回的是[1, H, W]格式的灰度图
        if img1.shape[1] == 3 and self.config.input_nc == 1:
            # RGB转灰度
            img1 = torch.mean(img1, dim=1, keepdim=True)
            img2 = torch.mean(img2, dim=1, keepdim=True)
        elif img1.shape[1] == 1 and self.config.input_nc == 3:
            # 灰度转RGB（复制3次）
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)
        
        # 前向传播
        with torch.no_grad():
            fused_img = self.model(img1, img2, strategy_type=self.config.fusion_strategy)
            
            # 计算损失
            total_loss, loss_dict = self.criterion(fused_img, img1, img2)
        
        return fused_img, loss_dict
    
    def compute_metrics_batch(self, fused_np, img1_np, img2_np):
        """
        计算评估指标
        
        Args:
            fused_np: 融合图像 (numpy数组)
            img1_np: 源图像1 (numpy数组)
            img2_np: 源图像2 (numpy数组)
            
        Returns:
            指标字典
        """
        metrics = {}
        
        try:
            # 基础指标
            metrics['EN'] = entropy(fused_np)
            metrics['SF'] = spatial_frequency(fused_np)
            metrics['SD'] = std_deviation(fused_np)
            
            # 互信息
            metrics['MI_S1'] = mutual_information(fused_np, img1_np)
            metrics['MI_S2'] = mutual_information(fused_np, img2_np)
            metrics['MI_Total'] = metrics['MI_S1'] + metrics['MI_S2']
            
            # SSIM（如果可用）
            try:
                from metrics import ssim_with_sources
                metrics['SSIM_S1'] = ssim_with_sources(fused_np, img1_np)
                metrics['SSIM_S2'] = ssim_with_sources(fused_np, img2_np)
                metrics['SSIM_Avg'] = (metrics['SSIM_S1'] + metrics['SSIM_S2']) / 2
            except:
                pass
            
            # Qabf（如果可用）
            try:
                from metrics import q_abf
                metrics['Qabf'] = q_abf(fused_np, img1_np, img2_np)
            except:
                pass
                
        except Exception as e:
            print(f"计算指标时出错: {e}")
        
        return metrics
    
    def test(self, test_loader, save_results=True):
        """
        完整测试流程
        
        Args:
            test_loader: 测试数据加载器
            save_results: 是否保存结果
            
        Returns:
            all_results: 所有测试结果列表
        """
        print(f"\n{'='*60}")
        print(f"开始测试")
        print(f"测试样本数: {len(test_loader.dataset)}")
        print(f"批次大小: {test_loader.batch_size}")
        print(f"{'='*60}\n")
        
        all_results = []
        all_losses = {
            'total': [],
            'ssim': [],
            'gradient': [],
            'intensity': [],
            'mi': []
        }
        
        start_time = time.time()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='测试中')
            
            for batch_idx, (img1, img2) in enumerate(pbar):
                batch_size = img1.shape[0]
                
                # 测试单个批次
                fused_img, loss_dict = self.test_single_batch(img1, img2, batch_idx)
                
                # 处理每个样本
                for i in range(batch_size):
                    # 转换为numpy
                    img1_np = self.tensor_to_numpy(img1[i])
                    img2_np = self.tensor_to_numpy(img2[i])
                    fused_np = self.tensor_to_numpy(fused_img[i])
                    
                    # 生成图像索引
                    img_idx = batch_idx * batch_size + i
                    
                    # 保存融合图像
                    if save_results and self.config.save_individual:
                        fused_path = os.path.join(
                            self.config.output_dir,
                            'fused_images',
                            f'fused_{img_idx:04d}.png'
                        )
                        self.save_image(fused_np, fused_path)
                    
                    # 保存对比图
                    if save_results and self.config.save_comparison:
                        comp_path = os.path.join(
                            self.config.output_dir,
                            'comparisons',
                            f'comparison_{img_idx:04d}.png'
                        )
                        self.save_comparison(img1_np, img2_np, fused_np, comp_path, img_idx)
                    
                    # 计算指标
                    metrics = {}
                    if self.config.compute_metrics:
                        metrics = self.compute_metrics_batch(fused_np, img1_np, img2_np)
                    
                    # 记录结果
                    result = {
                        'index': img_idx,
                        'loss_total': loss_dict.get('total', 0.0),
                        'loss_ssim': loss_dict.get('ssim', 0.0),
                        'loss_gradient': loss_dict.get('gradient', 0.0),
                        'loss_intensity': loss_dict.get('intensity', 0.0),
                        'loss_mi': loss_dict.get('mi', 0.0),
                        **metrics
                    }
                    all_results.append(result)
                    
                    # 累积损失
                    all_losses['total'].append(loss_dict.get('total', 0.0))
                    all_losses['ssim'].append(loss_dict.get('ssim', 0.0))
                    all_losses['gradient'].append(loss_dict.get('gradient', 0.0))
                    all_losses['intensity'].append(loss_dict.get('intensity', 0.0))
                    all_losses['mi'].append(loss_dict.get('mi', 0.0))
                
                # 更新进度条
                avg_loss = np.mean(all_losses['total'][-batch_size:])
                pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})
        
        # 计算平均指标
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"测试完成!")
        print(f"总测试时间: {total_time:.2f}秒")
        print(f"平均处理时间: {total_time/len(test_loader.dataset):.3f}秒/样本")
        print(f"{'='*60}\n")
        
        # 打印统计信息
        if all_results:
            print("平均损失:")
            for key, values in all_losses.items():
                if values:
                    print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
            
            if self.config.compute_metrics:
                print("\n平均指标:")
                metric_keys = [k for k in all_results[0].keys() 
                              if k not in ['index', 'loss_total', 'loss_ssim', 'loss_gradient', 
                                          'loss_intensity', 'loss_mi']]
                for key in metric_keys:
                    values = [r[key] for r in all_results if key in r and r[key] is not None]
                    if values:
                        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        
        return all_results
    
    def save_results(self, results, csv_path=None):
        """
        保存测试结果到CSV
        
        Args:
            results: 测试结果列表
            csv_path: CSV保存路径（默认使用配置中的路径）
        """
        if not results:
            print("没有结果可保存")
            return
        
        csv_path = csv_path or self.config.metrics_save_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        # 获取所有键
        keys = list(results[0].keys())
        
        # 保存CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"✓ 测试结果已保存到: {csv_path}")
        
        # 保存统计摘要
        summary_path = csv_path.replace('.csv', '_summary.json')
        summary = {}
        
        # 计算平均值和标准差
        for key in keys:
            values = [r[key] for r in results if key in r and r[key] is not None]
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 统计摘要已保存到: {summary_path}")


def main():
    """主函数"""
    print(f"\n{'='*60}")
    print(f"医学图像融合模型测试")
    print(f"{'='*60}\n")
    
    # 创建测试数据加载器
    print("加载测试数据...")
    test_loader = create_dataloaders_test(
        source1_test_img_path=args.source1_test_img_path,
        source2_test_img_path=args.source2_test_img_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"测试样本数: {len(test_loader.dataset)}\n")
    
    # 创建测试器
    tester = Tester(args)
    
    # 执行测试
    results = tester.test(test_loader, save_results=True)
    
    # 保存结果
    if results:
        tester.save_results(results)
    
    print(f"\n{'='*60}")
    print(f"所有结果保存在: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()