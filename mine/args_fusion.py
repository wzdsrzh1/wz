import os
import json
import torch

class args:
    """测试参数配置"""
    
    # 设备配置
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    
    # 模型配置
    input_nc = 1  # 输入通道数
    output_nc = 1  # 输出通道数
    
    # 测试数据路径
    source1_test_img_path = './dataset/CT-MRI/test/CT'
    source2_test_img_path = './dataset/CT-MRI/test/MRI'
    
    # 模型路径
    model_path = './checkpoints/best_model.pth'  # 默认使用最佳模型
    # 如果需要指定特定检查点
    checkpoint_path = None  # 例如: './checkpoints/checkpoint_epoch_10.pth'
    
    # 融合策略
    fusion_strategy = 'attention'  # 可选: 'attention', 'weighted', 'max', 'average'
    
    # 测试输出配置
    output_dir = './test_results'  # 融合结果保存目录
    save_comparison = True  # 是否保存对比图（源图像+融合图像）
    save_individual = True  # 是否保存每张融合图像
    
    # 评估指标配置
    compute_metrics = True  # 是否计算评估指标
    metrics_save_path = './test_results/metrics.csv'  # 指标保存路径
    
    # 数据加载配置
    image_size = (256, 256)  # 图像尺寸
    batch_size = 1  # 测试批次大小（通常为1以便逐张处理）
    num_workers = 1  # 数据加载进程数
    
    # 损失函数权重（用于测试时计算损失）
    w_ssim = 1.0
    w_gradient = 10.0
    w_intensity = 5.0
    w_perceptual = 1.0
    w_mi = 0.5
    use_perceptual = False

