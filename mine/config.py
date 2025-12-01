import os
import json
import torch

class Config:


    # 数据路径配置
    source1_img_paths = './dataset/CT-T1MRI/CT'
    source2_img_paths = './dataset/CT-T1MRI/T1-MRI'
    source1_train_img_path = './dataset/CT-T1MRI/CT'
    source2_train_img_path = './dataset/CT-T1MRI/T1-MRI'
    source1_test_img_path = './dataset/CT-MRI/test/CT'
    source2_test_img_path = './dataset/CT-MRI/test/MRI'
    source1_val_img_path = './dataset/CT-MRI/train/CT'
    source2_val_img_path = './dataset/CT-MRI/train/MRI'
    #模型选择
    model = 'medical'#可选medical/dense fusion

    # 数据加载配置
    image_size = (256, 256)
    batch_size = 6
    num_workers = 0

    # 模型配置
    input_nc = 1  # 输入通道数
    output_nc = 1  # 输出通道数

    # 训练配置
    epochs = 50  # 训练轮数
    lr = 1e-5  # 学习率
    weight_decay = 1e-5  # 权重衰减
    gradient_clip_norm = 1.0  # 梯度裁剪阈值

    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 损失函数权重配置
    w_ssim = 1.0  # SSIM损失权重
    w_gradient = 1.0  # 梯度损失权重
    w_intensity = 1.0  # 强度损失权重
    w_perceptual = 1.0  # 感知损失权重
    w_mi = 0  # 互信息损失权重
    w_deco = 1.5
    use_perceptual = False  # 是否使用感知损失（需要VGG预训练模型）
    #loss_4
    l_alpha = 3.0
    l_beta = 1.0
    l_gamma = 1.0
    mse_w = 0.01

    # 融合策略
    fusion_strategy = 'attention'  # 可选: 'attention', 'weighted', 'max', 'average'

    # 学习率调度器配置
    use_scheduler = True  # 是否使用学习率调度器
    scheduler = 'ReduceLROnPlateau'  # 可选: 'ReduceLROnPlateau', 'CosineAnnealingLR'
    scheduler_patience = 5  # ReduceLROnPlateau的耐心值
    scheduler_factor = 0.5  # 学习率衰减因子
    scheduler_min_lr = 1e-6  # 最小学习率

    # 保存和日志配置
    save_model_dir = './checkpoints'  # 模型保存目录
    save_loss_dir = './losses'  # 损失保存目录
    log_dir = './logs'  # 日志保存目录
    resume = None  # 恢复训练的检查点路径，如: './checkpoints/model.pth'

    # 训练日志配置
    log_interval = 10  # 每N个batch打印一次日志
    save_interval = 5  # 每N个epoch保存一次模型
    save_best = True  # 是否保存最佳模型

    # 验证配置
    val_interval = 5  # 每N个epoch验证一次

    #保存融合结果
    save_result = './result'
    #保存评价指标
    save_metrics = './metrics'
    #模型文件地址
    model_path = './checkpoints/best_model.pth'#默认加载最好的模型
    #是否保存每一场融合图像
    save_individual = True
    #评价指标保存地址
    metrics_save_path = './metrics.csv'
    #是否计算评价指标
    compute_metrics = True