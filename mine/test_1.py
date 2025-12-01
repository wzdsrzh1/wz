import os
import numpy as np
import time
from tqdm import tqdm
import torch
from model import MedicalFusion_net
from config import Config
from loss_2 import MedicalImageFusionLoss
from densefusion_net import DenseFuse_net
from dataload_1 import create_dataloaders_test
from metrics_1 import metrics_test
from PIL import Image
import csv
from loss_3 import ImageFusionLoss
from loss_4 import ImagenetLoss


class Tester:

    def __init__(self, config):
        self.config = config
        #创建保存目录
        os.makedirs(config.save_result,exist_ok = True)#保存融合图像
        os.makedirs(config.save_metrics,exist_ok = True)#保存评价指标
        self.device = torch.device(config.device)
        #加载模型
        self.model = self.load_model(model_path = self.config.model_path)
        #初始化损失函数
        self.loss = ImagenetLoss(
            l_alpha=self.config.l_alpha,
            l_beta=self.config.l_beta,
            l_gamma=self.config.l_gamma,
            mse_w = self.config.mse_w,
            ).to(self.device)

    def load_model(self,model_path):
        """
        加载训练好的模型
        Args:
            model_path:模型路径
        Returns:
            model:模型
        """
        #创建模型
        model = MedicalFusion_net(
            input_nc=self.config.input_nc,
            output_nc=self.config.output_nc
        ).to(self.device)
        #加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
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
        print(type(model))

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

    def test_batch(self,img_1,img_2,batch_idx):
        """
        测试单个batch
        Args:
            img_1:
            img_2:
        Returns:
            fused_img:融合图像
            loss_dict:各项损失的字典
        """
        img_1 = img_1.to(self.device)
        img_2 = img_2.to(self.device)

        with torch.no_grad():
            fused_img = self.model(img_1,img_2,strategy_type=self.config.fusion_strategy)

            # 计算损失
            total_loss, loss_dict = self.loss(fused_img , img_1 , img_2 )

        return fused_img, loss_dict

    def test(self,test_loader,save_results=True):
        """

        Args:
            test_loader:测试数据集加载
        """
        print(f"\n{'='*60}")
        print(f"开始测试")
        print(f"测试样本数: {len(test_loader.dataset)}")
        print(f"批次大小: {test_loader.batch_size}")
        print(f"{'='*60}\n")
        all_results = []
        total_loss = {
                      'total': [],
                      'ssim': [],
                      'gradient': [],
                      'contest': []
                      }
        start_time = time.time()
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='测试中')
            for batch_idx, (img1, img2) in enumerate(pbar):
                batch_size = img1.shape[0]
                # 测试单个批次
                fused_img, loss_dict = self.test_batch(img1, img2, batch_idx)
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
                            self.config.save_result,
                            'fused_images',
                            f'fused_{img_idx:04d}.png'
                        )
                        self.save_image(fused_np, fused_path)

                    metrics = metrics_test(device = self.device,fused_img=fused_np / 255, source1=img1_np / 255,source2=img2_np / 255)
                    result = {
                        'index': img_idx,
                        'loss_total': loss_dict.get('total_loss', 0.0),
                        'loss_ssim': loss_dict.get('ssim_loss', 0.0),
                        'loss_gradient': loss_dict.get('grad_loss', 0.0),
                        'loss_contest': loss_dict.get('contest_loss', 0.0),
                        **metrics
                    }
                    all_results.append(result)
                    total_loss['total'].append(loss_dict.get('total_loss', 0.0))
                    total_loss['ssim'].append(loss_dict.get('ssim_loss', 0.0))
                    total_loss['gradient'].append(loss_dict.get('grad_loss', 0.0))
                    total_loss['contest'].append(loss_dict.get('contest_loss', 0.0))

                    # 更新进度条
                avg_loss = np.mean(total_loss['total'][-batch_size:])
                pbar.set_postfix({'avg_loss': f'{avg_loss:.4f}'})

                # 计算平均指标
            total_time = time.time() - start_time
            print(f"\n{'=' * 60}")
            print(f"测试完成!")
            print(f"总测试时间: {total_time:.2f}秒")
            if all_results:
                print("平均损失:")
                for key, values in total_loss.items():
                    if values:
                        print(f"  {key}: {np.mean(values):.4f} ± {np.std(values):.4f}")

                if self.config.compute_metrics:
                    print("\n平均指标:")
                    metric_keys = [k for k in all_results[0].keys()
                                   if k not in ['index', 'loss_total', 'loss_ssim', 'loss_gradient',
                                                'loss_contest']]
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

def main():
    """主函数"""
    config = Config()
    print(f"\n{'=' * 60}")
    print(f"医学图像融合模型测试")
    print(f"{'=' * 60}\n")

    # 创建测试数据加载器
    print("加载测试数据...")
    test_loader = create_dataloaders_test(
        source1_test_img_path=config.source1_test_img_path,
        source2_test_img_path=config.source2_test_img_path,
        image_size=config.image_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )

    print(f"测试样本数: {len(test_loader.dataset)}\n")

    # 创建测试器
    tester = Tester(config)

    # 执行测试
    results = tester.test(test_loader, save_results=True)

    # 保存结果
    if results:
        tester.save_results(results)

    print(f"\n{'=' * 60}")
    print(f"所有结果保存在: {config.save_result}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()