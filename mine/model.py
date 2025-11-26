import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Convolution operation with Batch Normalization
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = self.bn(out)
            out = F.relu(out, inplace=True)
        return out


# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return out


# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


# Dense convolution unit with attention
class DenseConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DenseConv2d, self).__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.ca = ChannelAttention(out_channels)

    def forward(self, x):
        out = self.dense_conv(x)
        out = out * self.ca(out)
        out = torch.cat([x, out], 1)
        return out


# Enhanced Dense Block unit
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super(DenseBlock, self).__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels + out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels + out_channels_def * 2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

        # Add spatial attention at the end of dense block
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.denseblock(x)
        out = out * self.sa(out)
        return out


# Medical Image Fusion Network
class MedicalFusion_net(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super(MedicalFusion_net, self).__init__()
        denseblock = DenseBlock
        nb_filter = [32, 80, 64, 32, 16]  # Increased capacity
        kernel_size = 3
        stride = 1

        # Multi-scale encoder
        self.conv1 = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.DB1 = denseblock(nb_filter[0], kernel_size, stride)

        # Additional encoding layer for deeper features
        self.conv_mid = ConvLayer(nb_filter[1], nb_filter[1], kernel_size, stride)

        # Fusion attention module
        self.fusion_ca = ChannelAttention(nb_filter[1])
        self.fusion_sa = SpatialAttention()

        # decoder with skip connections support
        self.conv2 = ConvLayer(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.conv3 = ConvLayer(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.conv4 = ConvLayer(nb_filter[3], nb_filter[4], kernel_size, stride)
        self.conv5 = ConvLayer(nb_filter[4], output_nc, kernel_size, stride, is_last=True)
        self.output_activation = nn.Sigmoid()

    def encoder(self, input):
        x1 = self.conv1(input)
        x_DB = self.DB1(x1)
        x_mid = self.conv_mid(x_DB)
        return [x_mid]

    def fusion(self, en1, en2, strategy_type='attention'):
        """
        Enhanced fusion strategy for medical images
        Args:
            en1: encoded features from modality 1 (e.g., PET, CT)
            en2: encoded features from modality 2 (e.g., MRI, SPECT)
            strategy_type: fusion strategy ('attention', 'weighted', 'max')
        """
        if strategy_type == 'attention':
            # Channel attention fusion
            ca1 = self.fusion_ca(en1[0])
            ca2 = self.fusion_ca(en2[0])
            ca_weights = torch.softmax(torch.cat([ca1, ca2], dim=1), dim=1)
            ca_w1, ca_w2 = torch.chunk(ca_weights, 2, dim=1)

            # Spatial attention fusion
            sa1 = self.fusion_sa(en1[0])
            sa2 = self.fusion_sa(en2[0])

            # Combined attention fusion
            f_0 = (en1[0] * ca_w1 * sa1) + (en2[0] * ca_w2 * sa2)

        elif strategy_type == 'weighted':
            # Adaptive weighted fusion based on local energy
            w1 = torch.mean(torch.abs(en1[0]), dim=1, keepdim=True)
            w2 = torch.mean(torch.abs(en2[0]), dim=1, keepdim=True)
            w_sum = w1 + w2 + 1e-8
            f_0 = (en1[0] * w1 + en2[0] * w2) / w_sum

        elif strategy_type == 'max':
            # Maximum selection strategy
            f_0 = torch.max(en1[0], en2[0])

        else:  # default: average
            f_0 = (en1[0] + en2[0]) / 2

        return [f_0]

    def decoder(self, f_en):
        x2 = self.conv2(f_en[0])
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        output = self.conv5(x4)
        output = self.output_activation(output)
        return [output]

    def forward(self, img1, img2, strategy_type='attention'):
        """
        Complete forward pass for two-modality fusion
        """
        en1 = self.encoder(img1)
        en2 = self.encoder(img2)
        f = self.fusion(en1, en2, strategy_type)
        output = self.decoder(f)
        return output[0]


# Example usage
if __name__ == "__main__":
    # Create model
    model = MedicalFusion_net(input_nc=1, output_nc=1)

    # Example: PET and MRI fusion
    pet_image = torch.randn(1, 1, 256, 256)  # Batch=1, Channel=1, H=256, W=256
    mri_image = torch.randn(1, 1, 256, 256)

    # Fusion
    fused_image = model(pet_image, mri_image, strategy_type='attention')
    print(f"Fused image shape: {fused_image.shape}")

    # Model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")