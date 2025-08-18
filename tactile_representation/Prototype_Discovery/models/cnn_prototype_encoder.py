"""
CNN原型编码器模型 - 用于触觉力数据的原型权重计算
输入形状: (3, 20, 20) - 三通道触觉力图像
输出: 原型权重 (B, num_prototypes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out

class TactilePrototypeEncoder(nn.Module):
    """触觉原型编码器"""
    def __init__(self, in_channels=3, num_prototypes=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, num_prototypes)
        )

    def forward(self, x):
        weights = self.encoder(x)
        weights = F.softmax(weights, dim=-1)  # 归一化权重
        return weights

# 便利函数
def create_tactile_prototype_encoder(config):
    """根据配置创建触觉原型编码器"""
    return TactilePrototypeEncoder(
        in_channels=config.get('in_channels', 3),
        num_prototypes=config.get('num_prototypes', 16)
    )
