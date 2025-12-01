from pytorch_ssim import ssim,gradient,Fusionloss
import torch
from config import Config as c
import torch.nn as nn
import torch

def gray_to_ycbcr(gray_tensor):
    """
    将单通道灰度图像转换为3通道YCbCr图像
    Args:
        gray_tensor: [B, 1, H, W] 或 [1, H, W]，值范围[0,1]
    Returns:
        ycbcr_tensor: [B, 3, H, W] 或 [3, H, W]，YCbCr格式
        Y, Cb, Cr: 分离的通道
    """
    # 确保是浮点类型
    gray_tensor = gray_tensor.float()

    # Y通道就是灰度图像本身
    Y = gray_tensor.clone()

    # 对于灰度图像，Cb和Cr通道设为中性值（128/255 = 0.5）
    # 因为灰度图像没有颜色信息
    Cb = torch.full_like(Y, 0.5)
    Cr = torch.full_like(Y, 0.5)

    # 拼接成3通道YCbCr图像
    if len(Y.shape) == 3:  # [C, H, W] 格式
        ycbcr = torch.cat([Y, Cb, Cr], dim=0)
    else:  # [B, C, H, W] 格式
        ycbcr = torch.cat([Y, Cb, Cr], dim=1)

    return ycbcr, Y, Cb, Cr

def Grad_loss(output):
    # out_grad = torch.mean(gradient(output))
    # A_grad = torch.mean(gradient(output))
    # B_grad = torch.mean(gradient(output))
    out_grad = torch.mean(gradient(output), dim=[1, 2,3])
    out_grad = 1 - torch.mean(out_grad / (out_grad + 1.0))
    return out_grad

def Ssim_loss(fused_img_Y,img1_Y,img2_Y):
    LOSS_SSIM = 1 - ssim(fused_img_Y, img1_Y, img2_Y)
    return LOSS_SSIM

def guide_loss(output, bicubic_image):
    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss = loss_fn(output, bicubic_image)
    return loss

def Context_loss(fused_img_Y,img1_Y,img2_Y,mse_w):
    #亮度损失
    LOSS_CONTEXT = guide_loss(fused_img_Y, img2_Y) + mse_w * guide_loss(fused_img_Y, img1_Y)
    return LOSS_CONTEXT

class ImagenetLoss(nn.Module):
    def __init__(self,l_alpha = 1.0,l_beta = 1.0,mse_w = 1.0,l_gamma = 1.0):
        super(ImagenetLoss,self).__init__()
        self.l_alpha = l_alpha
        self.l_beta = l_beta
        self.mse_w = mse_w
        self.l_gamma = l_gamma
        self.Ssim_loss = Ssim_loss
        self.grad_loss = Grad_loss
        self.Context_loss = Context_loss

    def forward(self, fused_img, img1, img2):
        loss_dict = {}
        total_loss = 0
        fused_img_ycbcr,fused_img_Y,fused_img_Cb,fused_img_Cr = gray_to_ycbcr(fused_img)
        img1_ycbcr,img1_Y,img1_Cb,img1_Cr = gray_to_ycbcr(img1)
        img2_ycbcr,img2_Y,img2_Cb,img2_Cr = gray_to_ycbcr(img2)

        ssim_loss = self.Ssim_loss(fused_img_Y, img1_Y, img2_Y)
        loss_dict['ssim_loss'] = ssim_loss.item()

        grad_loss = self.grad_loss(fused_img)
        loss_dict['grad_loss'] = grad_loss.item()

        contest_loss = self.Context_loss(fused_img, img1_Y, img2_Y,mse_w = self.mse_w)
        loss_dict['contest_loss'] = contest_loss.item()

        total_loss = self.l_alpha * ssim_loss +self.l_beta * grad_loss + self.l_gamma * contest_loss
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict


