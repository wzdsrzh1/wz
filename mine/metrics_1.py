import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cv2
import os
import pandas as pd
from PIL import Image
from openpyxl import load_workbook
import warnings


warnings.filterwarnings('ignore')

# 设置中文显示
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class FusionMetrics:
    """多模态图像融合评价指标计算"""

    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def to_tensor(self, img):
        """将图像转换为GPU张量"""
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        if len(img.shape) == 2:
            img = img.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif len(img.shape) == 3:
            if img.shape[2] == 3:  # RGB转灰度
                img = torch.mean(img, dim=2)
            img = img.unsqueeze(0).unsqueeze(0)
        return img.to(self.device)

    def sobel_edge(self, img):
        """使用Sobel算子提取边缘特征"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        gradient = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        return gradient, grad_x, grad_y

    def calculate_entropy(self, img, bins=256):
        """计算图像熵"""
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * (bins - 1)).long()
        img_flat = img_norm.flatten()

        # 计算直方图
        hist = torch.zeros(bins, device=self.device)
        for i in range(len(img_flat)):
            hist[img_flat[i]] += 1

        # 归一化为概率
        prob = hist / hist.sum()

        # 计算熵
        entropy = 0.0
        for p in prob:
            if p > 0:
                entropy -= p * torch.log2(p)

        return entropy.item()

    def calculate_MI(self, img1, img2, bins=256):
        """计算互信息 (Mutual Information)"""
        img1_norm = ((img1 - img1.min()) / (img1.max() - img1.min() + 1e-8) * (bins - 1)).long()
        img2_norm = ((img2 - img2.min()) / (img2.max() - img2.min() + 1e-8) * (bins - 1)).long()

        img1_flat = img1_norm.flatten()
        img2_flat = img2_norm.flatten()

        # 计算联合直方图
        joint_hist = torch.zeros(bins, bins, device=self.device)
        for i in range(len(img1_flat)):
            joint_hist[img1_flat[i], img2_flat[i]] += 1

        joint_prob = joint_hist / joint_hist.sum()
        prob1 = joint_prob.sum(dim=1)
        prob2 = joint_prob.sum(dim=0)

        # 计算互信息
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_prob[i, j] > 0:
                    mi += joint_prob[i, j] * torch.log2(
                        joint_prob[i, j] / (prob1[i] * prob2[j] + 1e-10) + 1e-10
                    )

        return mi.item()

    def calculate_FMI(self, fusion_img, modality_A, modality_B):
        """
        计算特征互信息 (Feature Mutual Information)
        FMI = MI(edge_F, edge_A) + MI(edge_F, edge_B)
        """
        edge_f, _, _ = self.sobel_edge(fusion_img)
        edge_a, _, _ = self.sobel_edge(modality_A)
        edge_b, _, _ = self.sobel_edge(modality_B)

        mi_fa = self.calculate_MI(edge_f, edge_a)
        mi_fb = self.calculate_MI(edge_f, edge_b)

        h_a = self.calculate_entropy(edge_a)
        h_b = self.calculate_entropy(edge_b)
        h_f = self.calculate_entropy(edge_f)

        # 归一化互信息 NMI = 2*MI / (H1 + H2)
        nmi_fa = 2 * mi_fa / (h_f + h_a + 1e-10) if (h_f + h_a) > 0 else 0
        nmi_fb = 2 * mi_fb / (h_f + h_b + 1e-10) if (h_f + h_b) > 0 else 0

        # FMI为两个归一化互信息的平均值
        fmi = (nmi_fa + nmi_fb) / 2
        return max(0, min(1, fmi))  # 确保在[0,1]范围内

    def calculate_Qabf(self, fusion_img, modality_A, modality_B):
        """
        计算基于梯度的融合质量指标 Qabf
        综合考虑梯度幅值和方向的相似性
        """
        grad_f, gx_f, gy_f = self.sobel_edge(fusion_img)
        grad_a, gx_a, gy_a = self.sobel_edge(modality_A)
        grad_b, gx_b, gy_b = self.sobel_edge(modality_B)

        # 梯度幅值相似性
        C = 1e-4
        Qg_af = (2 * grad_a * grad_f + C) / (grad_a ** 2 + grad_f ** 2 + C)
        Qg_bf = (2 * grad_b * grad_f + C) / (grad_b ** 2 + grad_f ** 2 + C)

        # 梯度方向相似性
        numerator_a = torch.abs(gx_a * gx_f + gy_a * gy_f)
        denominator_a = torch.sqrt((gx_a ** 2 + gy_a ** 2) * (gx_f ** 2 + gy_f ** 2)) + C
        Qa_af = numerator_a / denominator_a

        numerator_b = torch.abs(gx_b * gx_f + gy_b * gy_f)
        denominator_b = torch.sqrt((gx_b ** 2 + gy_b ** 2) * (gx_f ** 2 + gy_f ** 2)) + C
        Qa_bf = numerator_b / denominator_b

        # 综合质量指数
        Q_af = Qg_af * Qa_af
        Q_bf = Qg_bf * Qa_bf

        # 相对显著性权重
        w_a = grad_a / (grad_a + grad_b + 1e-8)
        w_b = grad_b / (grad_a + grad_b + 1e-8)

        # 加权求和
        Qabf = (w_a * Q_af + w_b * Q_bf).mean()

        return Qabf.item()

    def calculate_SSIM(self, img1, img2, window_size=11):
        """
        计算结构相似性指数 SSIM
        SSIM = [l(x,y)]·[c(x,y)]·[s(x,y)]
        """
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # 创建高斯窗口
        sigma = 1.5
        gauss = torch.exp(-torch.arange(-window_size // 2 + 1, window_size // 2 + 1,
                                        dtype=torch.float32, device=self.device) ** 2 / (2 * sigma ** 2))
        window = (gauss / gauss.sum()).view(1, 1, 1, -1) * \
                 (gauss / gauss.sum()).view(1, 1, -1, 1)

        # 计算局部均值
        mu1 = F.conv2d(img1, window, padding=window_size // 2)
        mu2 = F.conv2d(img2, window, padding=window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # 计算局部方差和协方差
        sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu1_mu2

        # 计算SSIM
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean().item()

    def calculate_SSIR(self, fusion_img, modality_A, modality_B):
        """
        计算光谱保真度指标 SSIR
        使用相关系数衡量光谱相似性
        """

        def correlation(x, y):
            x_mean = x.mean()
            y_mean = y.mean()
            numerator = ((x - x_mean) * (y - y_mean)).sum()
            denominator = torch.sqrt(((x - x_mean) ** 2).sum() * ((y - y_mean) ** 2).sum())
            return (numerator / (denominator + 1e-8)).item()

        corr_a = correlation(fusion_img, modality_A)
        corr_b = correlation(fusion_img, modality_B)

        ssir = (corr_a + corr_b) / 2
        return ssir

    def calculate_PSNR(self, fusion, source_A, source_B):
        """
        计算峰值信噪比 (Peak Signal-to-Noise Ratio)
        PSNR = 10 * log10(MAX^2 / MSE)
        返回与两个源图像的平均PSNR
        """

        def psnr_single(img1, img2):
            mse = torch.mean((img1 - img2) ** 2)
            if mse < 1e-10:
                return 100.0
            max_pixel = 1.0  # 图像已归一化到[0,1]
            psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
            return psnr.item()

        psnr_a = psnr_single(fusion, source_A)
        psnr_b = psnr_single(fusion, source_B)

        # 返回平均PSNR
        return (psnr_a + psnr_b) / 2

    def evaluate_all(self, fusion_img, modality_A, modality_B):
        """计算所有评价指标"""
        fusion_img = self.to_tensor(fusion_img)
        modality_A = self.to_tensor(modality_A)
        modality_B = self.to_tensor(modality_B)

        metrics = {}

        print("正在计算 FMI...")
        metrics['FMI'] = self.calculate_FMI(fusion_img, modality_A, modality_B)

        print("正在计算 Qabf...")
        metrics['Qabf'] = self.calculate_Qabf(fusion_img, modality_A, modality_B)

        print("正在计算 SSIM...")
        ssim_a = self.calculate_SSIM(fusion_img, modality_A)
        ssim_b = self.calculate_SSIM(fusion_img, modality_B)
        metrics['SSIM'] = (ssim_a + ssim_b) / 2

        print("正在计算 SSIR...")
        metrics['SSIR'] = self.calculate_SSIR(fusion_img, modality_A, modality_B)

        print("正在计算 PSNR...")
        metrics['PSNR'] = self.calculate_PSNR(fusion_img, modality_A, modality_B)

        return metrics

def save_to_list(save_list,metrics):
    save_list.append(metrics)
    return save_list

def save_to_excel(save_path,save_list):
    path_excel = os.path.join(save_path, 'PET-MRI' + '.xlsx')
    file_excel = os.path.exists(path_excel)
    df = pd.DataFrame(save_list)
    if not file_excel:
        #用现有列表直接创建excel
        df.to_excel(path_excel, index=False,engine='openpyxl')
    else:
        #读取现有数据
        existing_df = pd.read_excel(path_excel)
        # 合并数据
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        # 保存合并后的数据
        combined_df.to_excel(path_excel, index=False, engine='openpyxl')
    print(f'结果已保存在{path_excel}')

def visualize_results(modality_A, modality_B, fusion_img, metrics, save_path='fusion_evaluation.png'):
    """可视化评价结果 - 表格版本"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 显示图像
    ax1 = fig.add_subplot(gs[0, 0])
    if len(modality_A.shape) == 3:
        ax1.imshow(cv2.cvtColor(modality_A, cv2.COLOR_BGR2RGB))
    else:
        ax1.imshow(modality_A, cmap='gray')
    ax1.set_title('模态 A ', fontsize=13, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    if len(modality_B.shape) == 3:
        ax2.imshow(cv2.cvtColor(modality_B, cv2.COLOR_BGR2RGB))
    else:
        ax2.imshow(modality_B, cmap='gray')
    ax2.set_title('模态 B ', fontsize=13, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    if len(fusion_img.shape) == 3:
        ax3.imshow(cv2.cvtColor(fusion_img, cv2.COLOR_BGR2RGB))
    else:
        ax3.imshow(fusion_img, cmap='gray')
    ax3.set_title('融合图像', fontsize=13, fontweight='bold')
    ax3.axis('off')

    # 显示评价指标 - 表格形式
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('tight')
    ax4.axis('off')

    # 准备表格数据
    metrics_names = list(metrics.keys())
    metrics_values = [f'{v:.4f}' for v in metrics.values()]

    # 指标说明
    metrics_desc = {
        'FMI': '特征互信息 (越大越好)',
        'Qabf': '梯度质量指标 (越大越好)',
        'SSIM': '结构相似性 (越接近1越好)',
        'SSIR': '光谱保真度 (越接近1越好)',
        'PSNR': '峰值信噪比 (越大越好)'
    }

    metrics_descriptions = [metrics_desc.get(name, '') for name in metrics_names]

    # 创建表格数据
    table_data = []
    for name, value, desc in zip(metrics_names, metrics_values, metrics_descriptions):
        table_data.append([name, value, desc])

    # 创建表格
    table = ax4.table(cellText=table_data,
                      colLabels=['指标名称', '指标值', '说明'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2, 0.2, 0.6])

    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 设置表头样式
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4ECDC4')
        cell.set_text_props(weight='bold', color='white', fontsize=12)

    # 设置数据行样式
    colors = ['#FFE5E5', '#E5F5FF', '#E5FFE5', '#FFF5E5']
    for i in range(1, len(table_data) + 1):
        color_idx = (i - 1) % len(colors)
        for j in range(3):
            cell = table[(i, j)]
            cell.set_facecolor(colors[color_idx])
            if j == 1:  # 指标值列加粗
                cell.set_text_props(weight='bold', fontsize=11)

    plt.suptitle('多模态图像融合评价分析', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存到: {save_path}")
    plt.close()

def load_image(path):
    """加载图像"""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"无法加载图像: {path}")

    # 归一化到[0, 1]
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0

    return img

def metrics_test(device,fused_img,source1,source2):

    evaluator = FusionMetrics(device = device)
    print('-'*30,'开始计算评价指标','-'*30)
    metrics = evaluator.evaluate_all(fused_img,source1 , source2)
    return metrics





# ==================== 主程序 ====================
if __name__ == "__main__":
    """
    使用说明:
    1. 准备三张图像: 源图像A、源图像B、融合图像
    2. 修改下面的图像路径
    3. 运行代码即可得到评价结果和可视化
    """

    # ========== 在这里修改您的图像路径 ==========
    #path_source_A = './Dataset/testsets/PET-MRI/MRI/41041.png'  # 模态1图像路径
    #path_source_B = './Dataset/testsets/PET-MRI/PET/41041.png'  # 模态2图像路径
    #path_fusion = './results/SwinFusion_PET-MRI/41041.png'  # 融合图像路径
    #模态A图像路径
    path_modality_A = './Dataset/testsets/PET-MRI/MRI'
    #模态B图像路径
    path_modality_B ='./Dataset/testsets/PET-MRI/PET'
    #融合图像路径
    path_fusion_img = './results/SwinFusion_PET-MRI'
    #结果保存地址
    save_path = './results/metrics/PET-MRI'
    #空列表用于存放计算结果（字典）
    save_list =[]
    # =========================================
    file_name = os.listdir(path_fusion_img)
    length = len(file_name)

    print("=" * 60)
    print("多模态图像融合评价指标计算系统")
    print("=" * 60)
    for i in range(length):
        # 加载图像
        print(f"\n正在加载图像{file_name[i]}")
        try:
            img_name = file_name[i]
            modality_A = load_image(os.path.join(path_modality_A, img_name))
            modality_B = load_image(os.path.join(path_modality_B, img_name))
            fusion_img = load_image(os.path.join(path_fusion_img, img_name))
            print(f"模态A尺寸: {modality_A.shape}")
            print(f"模态B尺寸: {modality_B.shape}")
            print(f"融合图像尺寸: {fusion_img.shape}")
        except Exception as e:
            print(f"加载图像失败: {e}")
            print("\n请确保图像路径正确，支持的格式: PNG, JPG, BMP, TIFF等")
            exit(1)

        # 初始化评价器
        print("\n初始化GPU评价器...")
        evaluator = FusionMetrics(device='cuda')

        # 计算所有指标
        print("\n" + "=" * 60)
        print("开始计算融合评价指标...")
        print("=" * 60 + "\n")

        metrics = evaluator.evaluate_all(fusion_img, modality_A, modality_B)
        save_list = save_to_list(save_list, metrics)
        # 打印结果
        print("\n" + "=" * 60)
        print("评价指标计算结果:")
        print("=" * 60)
        for metric, value in metrics.items():
            if metric == 'FMI':
                print(f"{metric:10s}: {value:.6f}  (特征互信息，越大越好)")
            elif metric == 'Qabf':
                print(f"{metric:10s}: {value:.6f}  (梯度质量指标，越大越好)")
            elif metric == 'SSIM':
                print(f"{metric:10s}: {value:.6f}  (结构相似性，越接近1越好)")
            elif metric == 'SSIR':
                print(f"{metric:10s}: {value:.6f}  (光谱保真度，越接近1越好)")
            elif metric == 'PSNR':
                print(f"{metric:10s}: {value:.6f}  (峰值信噪比，越大越好)")
        print("=" * 60)

        # 生成可视化
        print("\n正在生成可视化结果...")
        visualize_results(modality_A, modality_B, fusion_img, metrics,
                          save_path=os.path.join(save_path, img_name))
        print("=" * 60)
    save_to_excel(save_path, save_list)
    print("\n评价完成!")
    print("=" * 60)