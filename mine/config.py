import os
import json
import torch

class Config:
    # 数据路径配置
    pet_train_img_path = 'D:/dataset/AANLIB/MyDatasets/PET-MRI/train/PET'
    mri_train_img_path = 'D:/dataset/AANLIB/MyDatasets/PET-MRI/train/MRI'
    pet_test_img_path = 'D:/dataset/AANLIB/MyDatasets/PET-MRI/test/PET'
    mri_test_img_path = 'D:/dataset/AANLIB/MyDatasets/PET-MRI/test/MRI'

    #模型选择
    model = 'medical'#可选medical/dense fusion

    # 数据加载配置
    image_size = (256, 256)
    batch_size = 6
    num_workers = 1

    # 模型配置
    input_nc = 3  # 输入通道数
    output_nc = 3  # 输出通道数

    # 训练配置
    epochs = 100  # 训练轮数
    lr = 1e-4  # 学习率
    weight_decay = 1e-5  # 权重衰减
    gradient_clip_norm = 1.0  # 梯度裁剪阈值

    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 损失函数权重配置
    w_ssim = 1.0  # SSIM损失权重
    w_gradient = 10.0  # 梯度损失权重
    w_intensity = 5.0  # 强度损失权重
    w_perceptual = 1.0  # 感知损失权重
    w_mi = 0.5  # 互信息损失权重
    use_perceptual = False  # 是否使用感知损失（需要VGG预训练模型）

    # 融合策略
    fusion_strategy = 'attention'  # 可选: 'attention', 'weighted', 'max', 'average'

    # 学习率调度器配置
    use_scheduler = True  # 是否使用学习率调度器
    scheduler_type = 'ReduceLROnPlateau'  # 可选: 'ReduceLROnPlateau', 'CosineAnnealingLR'
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
    save_interval = 50  # 每N个epoch保存一次模型
    save_best = True  # 是否保存最佳模型

    # 验证配置
    val_interval = 5  # 每N个epoch验证一次