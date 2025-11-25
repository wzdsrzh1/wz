import os
import time
from tqdm import tqdm
import torch
from torch.optim import Adam
from model import MedicalFusion_net
from config import Config
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from loss import MedicalImageFusionLoss
from densefusion_net import DenseFuse_net
from dataload_1 import create_dataloaders_train
import json

class Trainer:
    def __init__(self,config):
        self.config = config
        self.device = torch.device(config.device)
        #创建保存目录
        os.makedirs(config.save_model_dir, exist_ok=True)
        os.makedirs(config.save_loss_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
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
                                                   min_lr=config.scheduler_min_lr
                                                   )
            else :
                self.scheduler = None
        else:
            self.scheduler = None
        #训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')#初始化验证损失为+∞
        self.current_epoch = 0#记录当前轮次数
        #是否从检查点恢复训练
        if config.resume and os.path.exists(config.resume):
            self.load_checkpoint(config.resume)

    def train_epoch(self,train_loader):
        """
        每个epoch的训练流程

        Args:
        train_loader:训练数据加载器
        Returns:
            train_loss: 平均损失
            total_loss_dict: 各项损失的字典
        """
        self.model.train()
        train_loss = 0.0
        total_loss_dict = {}
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch} /{self.config.epochs}')

        for batch_idx,batch in enumerate(pbar):
            img1,img2=batch
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            self.optimizer.zero_grad()
            if self.config.model == 'medical':
                fused_img = self.model(img1, img2, strategy_type=self.config.fusion_strategy)
            elif self.config.model == 'dense fusion':
                fused_img = self.model(img1, img2)
            total_loss,loss_dict = self.loss(fused_img, img1, img2)
            total_loss.backward()
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.gradient_clip_norm
                )
            self.optimizer.step()
            #累计损失
            train_loss += total_loss.item()
            for key,value in loss_dict.items():
                if key in total_loss_dict:
                    total_loss_dict[key] += value
                else:
                    total_loss_dict[key] = value
                #更新进度条
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
            for key in total_loss_dict:
                total_loss_dict[key] /= num_batches

        return train_loss, total_loss_dict

    def validate(self,val_loader):
        """
        验证模型

        Args:
            val_loader:验证数据加载
        Returns:
            val_loss:验证损失
            total_loss_dict:各项损失的字典
        """
        self.model.eval()
        val_loss = 0.0
        total_loss_dict = {}
        num_batches = len(val_loader)
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch_idx, (img1, img2) in enumerate(pbar):
                # 将数据移到设备
                img1 = img1.to(self.device)
                img2 = img2.to(self.device)
                fused_img = self.model(img1, img2, strategy_type=self.config.fusion_strategy)
                total_loss, loss_dict = self.loss(fused_img, img1, img2)
                #累计损失
                val_loss += total_loss.item()
                for key,value in loss_dict.items():
                    if key in total_loss_dict:
                        total_loss_dict[key] += value
                    else:
                        total_loss_dict[key] = value

        #计算平均损失
        val_loss /= num_batches
        for key in total_loss_dict:
            total_loss_dict[key] /= num_batches
        return val_loss, total_loss_dict

    def save_checkpoint(self, is_best=False):
        """
        保存检查点

        Args:
            is_best:是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': {
                'input_nc': self.config.input_nc,
                'output_nc': self.config.output_nc,
                'fusion_strategy': self.config.fusion_strategy,
                'batch_size' :self.config.batch_size
            },
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        checkpoint_path = os.path.join(
            self.config.save_model_dir,
            f'checkpoint_epoch_{self.current_epoch + 1}.pth'  # 包含epoch编号
        )
        torch.save(checkpoint, checkpoint_path)
        # 保存最新模型 - 始终覆盖
        latest_path = os.path.join(self.config.save_model_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        # 保存最佳模型 - 条件性保存
        if is_best and self.config.save_best:
            best_path = os.path.join(self.config.save_model_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        """
        加载模型检查点

        Args:
            checkpoint_path: 检查点文件路径
        """
        print(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"已加载检查点，从epoch {self.current_epoch + 1}继续训练")

    def save_losses(self, train_loss, val_loss=None):
        """
        保存损失到文件

        Args:
            train_loss: 训练损失字典
            val_loss: 验证损失字典（可选）
        """
        # 保存训练损失
        loss_file = os.path.join(self.config.save_loss_dir, 'train_losses.json')
        self.train_losses.append({
            'epoch': self.current_epoch + 1,
            'losses': train_loss
        })

        # 保存验证损失
        if val_loss is not None:
            self.val_losses.append({
                'epoch': self.current_epoch + 1,
                'losses': val_loss
            })

        # 保存到JSON文件
        with open(loss_file, 'w') as f:
            json.dump({
                'train': self.train_losses,
                'val': self.val_losses
            }, f, indent=2)

    def train(self, train_loader, val_loader=None):
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        print(f"\n{'=' * 60}")
        print(f"开始训练 - 医学图像融合模型")
        print(f"设备: {self.device}")
        print(f"训练轮数: {self.config.epochs}")
        print(f"批次大小: {self.config.batch_size}")
        print(f"学习率: {self.config.lr}")
        print(f"融合策略: {self.config.fusion_strategy}")
        print(f"{'=' * 60}\n")
        start_time = time.time()
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            train_loss,loss_dict = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            val_loss = None
            val_loss_dict = None
            if val_loader is not None and (epoch + 1) % self.config.val_interval == 0:
                val_loss, val_loss_dict = self.validate(val_loader)
                self.val_losses.append(val_loss)

                # 更新最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    print(f"\n✓ 发现更好的模型 (验证损失: {val_loss:.4f})")
            else:
                is_best = False
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    if val_loss is not None:
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step(train_loss)
                else:
                    self.scheduler.step()
            # 打印epoch总结
            print(f"\nEpoch {epoch + 1}/{self.config.epochs} 总结:")
            print(f"  训练损失: {train_loss:.4f}")
            for key, value in loss_dict.items():
                print(f"    - {key}: {value:.4f}")

            if val_loss is not None:
                print(f"  验证损失: {val_loss:.4f}")
                for key, value in val_loss_dict.items():
                    print(f"    - {key}: {value:.4f}")

            # 保存损失
            self.save_losses(loss_dict, val_loss_dict)

            # 保存检查点
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            print("-" * 60)

        total_time = time.time() - start_time
        print(f'训练结束，总用时{total_time:.2f}s')
        print(f"模型保存在: {self.config.save_model_dir}")

def main():
    config = Config()

    train_loader,val_loader = create_dataloaders_train(
                source1_train_img_path=config.source1_train_img_path,
                source2_train_img_path=config.source2_train_img_path,
                source1_val_img_path=config.source1_val_img_path,
                source2_val_img_path=config.source2_val_img_path,
                image_size=config.image_size,
                batch_size=config.batch_size,
                num_workers=config.num_workers)

    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main()