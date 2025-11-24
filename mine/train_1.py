import os
import sys
import time
import numpy as np
from jinja2 import optimizer
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from torch.autograd import Variable
import utils
from model import MedicalFusion_net
from config import Config
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from loss import MedicalImageFusionLoss
from densefusion_net import DenseFuse_net
from dataload_1 import create_dataloaders_train

class Trainer:
    def __init__(self,config):
        self.config = config
        self.device = torch.device(config.device)
        #初始化损失函数
        self.loss = MedicalImageFusionLoss(
            w_ssim=config.w_ssim,
            w_gradient=config.w_gradient,
            w_intensity=config.w_intensity,
            w_mi=Config.w_mi,
            use_perceptual=config.use_perceptual,
            ).to(self.device)
        #初始化模型
        if config.model == 'medical':
            self.model = MedicalFusion_net(input_nc = config.input_nc,
                                       output_nc = config.output_nc
                                       ).to(self.device)
        elif config.model == 'dense fusion':
            self.model = DenseFuse_net(input_nc = config.input_nc,
                                       output_nc = config.output_nc
                                       ).to(self.device)
        #初始化优化器
        self.optimizer = Adam(self.model.parameters(),
                              lr=config.lr,
                              weight_decay=config.weight_decay)
        #学习率调度器
        if config.use_scheduler:
            if config.scheduler == "CosineAnnealing":
                self.scheduler = CosineAnnealingLR(optimizer=self.optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.scheduler_min_lr)
            elif config.scheduler == "ReduceLROnPlateau":
                self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer,
                                                   mode='min',
                                                   factor=config.scheduler_factor,
                                                   patience=config.scheduler_patience,
                                                   verbose=True,
                                                   min_lr=config.scheduler_min_lr
                                                   )
            else :
                self.scheduler = None
        else:
            self.scheduler = None
        #训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        #是否从检查点恢复训练
        if config.resume and os.path.exists(config.resume):
            self.load_checkpoint(config.resume)

    def train_epoch(self,train_loader):
        """
        Args:
        train_loader:训练数据加载器

        Returns:
            train_loss: 平均损失
            loss_dict: 各项损失的字典
        """
        self.model.train()
        train_loss = 0.0
        loss_dict = {}
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch} /{self.config.epochs}')

        for batch_idx,batch in enumerate(pbar):
            source_1,source_2=batch
            source_1_img = source_1.to(self.device)
            source_2_img = source_2.to(self.device)
            self.optimizer.zero_grad()
            fused_img = self.model(source_1_img, source_2_img, strategy_type=self.config.fusion_strategy)
            total_loss,loss_dict = self.loss(fused_img, source_1_img, source_2_img)
            total_loss.backward()
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_norm
                )
            self.optimizer.step()
            train_loss += total_loss.item()
            for key,value in loss_dict.items():
                loss_dict[key] += value.item()

                if (batch_idx + 1) % self.config.log_interval == 0:
                    avg_loss = train_loss / (batch_idx + 1)
                    loss_str = ', '.join([f'{k}: {v / (batch_idx + 1):.4f}'
                                          for k, v in loss_dict.items() if k != 'total'])
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'details': loss_str
                    })

            # 计算平均损失
            train_loss /= num_batches
            for key in loss_dict:
                loss_dict[key] /= num_batches

        return train_loss, loss_dict
