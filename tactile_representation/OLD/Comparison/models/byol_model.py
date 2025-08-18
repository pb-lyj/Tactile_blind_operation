"""
BYOL (Bootstrap Your Own Latent) 模型实现，适配触觉力数据 (3, 20, 20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class ResNetBlock(nn.Module):
    """ResNet基础块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TactileEncoder(nn.Module):
    """触觉数据编码器（类似ResNet结构）"""
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, base_dim, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_dim)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet块
        self.layer1 = self._make_layer(base_dim, base_dim, 2, stride=1)
        self.layer2 = self._make_layer(base_dim, base_dim * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_dim * 2, base_dim * 4, 2, stride=2)
        
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 输出维度
        self.output_dim = base_dim * 4

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


class ProjectionHead(nn.Module):
    """投影头"""
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )

    def forward(self, x):
        return self.projection(x)


class PredictionHead(nn.Module):
    """预测头"""
    def __init__(self, input_dim, hidden_dim=512, output_dim=128):
        super().__init__()
        self.prediction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )

    def forward(self, x):
        return self.prediction(x)


class TactileBYOL(nn.Module):
    """
    触觉BYOL模型
    """
    def __init__(self, base_encoder=None, projection_dim=128, prediction_dim=128, 
                 hidden_dim=256, moving_average_decay=0.99):
        super().__init__()
        
        if base_encoder is None:
            base_encoder = TactileEncoder()
        
        # 在线网络
        self.online_encoder = base_encoder
        self.online_projector = ProjectionHead(
            self.online_encoder.output_dim, hidden_dim, projection_dim
        )
        self.online_predictor = PredictionHead(
            projection_dim, hidden_dim, prediction_dim
        )
        
        # 目标网络（通过指数移动平均更新）
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # 冻结目标网络
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
        
        self.moving_average_decay = moving_average_decay

    def forward(self, x1, x2):
        """
        前向传播
        Args:
            x1, x2: 同一输入的两个不同增强版本
        """
        # 在线网络
        online_repr_1 = self.online_encoder(x1)
        online_proj_1 = self.online_projector(online_repr_1)
        online_pred_1 = self.online_predictor(online_proj_1)
        
        online_repr_2 = self.online_encoder(x2)
        online_proj_2 = self.online_projector(online_repr_2)
        online_pred_2 = self.online_predictor(online_proj_2)
        
        # 目标网络
        with torch.no_grad():
            target_repr_1 = self.target_encoder(x1)
            target_proj_1 = self.target_projector(target_repr_1)
            
            target_repr_2 = self.target_encoder(x2)
            target_proj_2 = self.target_projector(target_repr_2)
        
        return {
            'online_pred_1': online_pred_1,
            'online_pred_2': online_pred_2,
            'target_proj_1': target_proj_1.detach(),
            'target_proj_2': target_proj_2.detach(),
            'online_repr_1': online_repr_1,
            'online_repr_2': online_repr_2
        }

    def encode(self, x):
        """编码单个输入"""
        return self.online_encoder(x)

    @torch.no_grad()
    def update_target_network(self):
        """更新目标网络参数"""
        def update_params(online_params, target_params):
            for online_param, target_param in zip(online_params, target_params):
                target_param.data = (
                    target_param.data * self.moving_average_decay +
                    online_param.data * (1.0 - self.moving_average_decay)
                )
        
        update_params(self.online_encoder.parameters(), self.target_encoder.parameters())
        update_params(self.online_projector.parameters(), self.target_projector.parameters())


class TactileBYOLLoss(nn.Module):
    """
    BYOL损失函数
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs):
        """
        计算BYOL损失
        """
        online_pred_1 = outputs['online_pred_1']
        online_pred_2 = outputs['online_pred_2']
        target_proj_1 = outputs['target_proj_1']
        target_proj_2 = outputs['target_proj_2']
        
        # 归一化
        online_pred_1 = F.normalize(online_pred_1, dim=1)
        online_pred_2 = F.normalize(online_pred_2, dim=1)
        target_proj_1 = F.normalize(target_proj_1, dim=1)
        target_proj_2 = F.normalize(target_proj_2, dim=1)
        
        # 计算损失
        loss_1 = 2 - 2 * (online_pred_1 * target_proj_2).sum(dim=1)
        loss_2 = 2 - 2 * (online_pred_2 * target_proj_1).sum(dim=1)
        
        total_loss = (loss_1 + loss_2).mean()
        
        return {
            'total_loss': total_loss,
            'byol_loss': total_loss
        }


class TactileBYOLClassifier(nn.Module):
    """
    基于BYOL的分类器
    """
    def __init__(self, byol_model, num_classes, freeze_byol=True):
        super().__init__()
        self.byol = byol_model
        if freeze_byol:
            for param in self.byol.parameters():
                param.requires_grad = False
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.byol.online_encoder.output_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 提取BYOL特征
        with torch.no_grad() if not self.training else torch.enable_grad():
            features = self.byol.encode(x)
        
        # 分类
        return self.classifier(features)


class TactileDataAugmentation:
    """
    触觉数据增强
    """
    def __init__(self, noise_std=0.1, rotation_angle=15, flip_prob=0.5):
        self.noise_std = noise_std
        self.rotation_angle = rotation_angle
        self.flip_prob = flip_prob

    def __call__(self, x):
        """
        应用数据增强
        Args:
            x: 输入张量 (C, H, W)
        Returns:
            增强后的两个版本
        """
        x1 = self.augment_single(x)
        x2 = self.augment_single(x)
        return x1, x2

    def augment_single(self, x):
        """对单个样本应用增强"""
        # 复制输入
        x_aug = x.clone()
        
        # 添加高斯噪声
        if torch.rand(1) > 0.5:
            noise = torch.randn_like(x_aug) * self.noise_std
            x_aug = x_aug + noise
        
        # 随机翻转
        if torch.rand(1) < self.flip_prob:
            x_aug = torch.flip(x_aug, dims=[1])  # 水平翻转
        
        if torch.rand(1) < self.flip_prob:
            x_aug = torch.flip(x_aug, dims=[2])  # 垂直翻转
        
        # 随机旋转（简化版，只做90度的倍数）
        if torch.rand(1) > 0.7:
            k = torch.randint(1, 4, (1,)).item()
            x_aug = torch.rot90(x_aug, k, dims=[1, 2])
        
        return x_aug


def create_tactile_byol(config=None):
    """
    创建触觉BYOL模型的工厂函数
    """
    if config is None:
        config = {
            'projection_dim': 128,
            'prediction_dim': 128,
            'hidden_dim': 256,
            'moving_average_decay': 0.99
        }
    
    encoder = TactileEncoder()
    model = TactileBYOL(base_encoder=encoder, **config)
    return model


def create_byol_loss():
    """
    创建BYOL损失函数的工厂函数
    """
    return TactileBYOLLoss()


def create_data_augmentation(config=None):
    """
    创建数据增强的工厂函数
    """
    if config is None:
        config = {
            'noise_std': 0.1,
            'rotation_angle': 15,
            'flip_prob': 0.5
        }
    
    return TactileDataAugmentation(**config)
