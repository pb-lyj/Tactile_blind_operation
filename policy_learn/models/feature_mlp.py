"""
Feature-MLP模型 - 基于预训练触觉特征的行为克隆
使用预训练的CNN自编码器提取左右手触觉特征，然后用MLP学习动作映射
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加路径以导入触觉表示模型
sys.path.append(os.path.join(os.path.dirname(__file__), '../../tactile_representation/Prototype_Discovery/models'))

try:
    from cnn_autoencoder import TactileCNNAutoencoder
except ImportError:
    print("警告: 无法导入 TactileCNNAutoencoder，将使用占位符")
    class TactileCNNAutoencoder(nn.Module):
        def __init__(self, in_channels=3, latent_dim=128):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, latent_dim)
            )
        
        def encode(self, x):
            return self.encoder(x)


class FeatureMLP(nn.Module):
    """
    基于预训练触觉特征的MLP策略网络
    
    输入: 左右手触觉特征 (256维 = 128 + 128)
    输出: delta_action (3维位置增量)
    """
    
    def __init__(self, 
                 feature_dim=128,  # 单手特征维度
                 action_dim=3,     # 输出动作维度
                 hidden_dims=[512, 512, 512],  # 隐藏层维度
                 dropout_rate=0.1,
                 pretrained_encoder_path=None):
        """
        Args:
            feature_dim: 单手触觉特征维度
            action_dim: 输出动作维度  
            hidden_dims: MLP隐藏层维度列表
            dropout_rate: Dropout比率
            pretrained_encoder_path: 预训练编码器权重路径
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # 加载预训练的触觉特征提取器
        self.tactile_encoder = TactileCNNAutoencoder(
            in_channels=3, 
            latent_dim=feature_dim
        )
        
        # 如果提供了预训练权重路径，加载权重
        if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
            print(f"加载预训练触觉编码器: {pretrained_encoder_path}")
            try:
                checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    self.tactile_encoder.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.tactile_encoder.load_state_dict(checkpoint)
                print("预训练权重加载成功")
            except Exception as e:
                print(f"权重加载失败: {e}, 使用随机初始化")
        else:
            print(f"警告: 未找到预训练权重文件 {pretrained_encoder_path}, 使用随机初始化")
        
        # 冻结特征提取器参数
        for param in self.tactile_encoder.parameters():
            param.requires_grad = False
        print(f"特征提取器参数已冻结")
        
        # 构建MLP网络
        # 输入维度: 左右手特征连接 = feature_dim * 2
        input_dim = feature_dim * 2
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # 使用LayerNorm而不是BatchNorm
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化MLP权重"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def extract_features(self, forces_l, forces_r):
        """
        提取左右手触觉特征
        
        Args:
            forces_l: 左手触觉数据 (B, 3, 20, 20)
            forces_r: 右手触觉数据 (B, 3, 20, 20)
            
        Returns:
            features: 连接的特征向量 (B, feature_dim * 2)
        """
        with torch.no_grad():  # 特征提取不需要梯度
            feature_l = self.tactile_encoder.encode(forces_l)  # (B, feature_dim)
            feature_r = self.tactile_encoder.encode(forces_r)  # (B, feature_dim)
        
        # 连接左右手特征
        features = torch.cat([feature_l, feature_r], dim=1)  # (B, feature_dim * 2)
        return features
    
    def forward(self, forces_l, forces_r):
        """
        前向传播
        
        Args:
            forces_l: 左手触觉数据 (B, 3, 20, 20)
            forces_r: 右手触觉数据 (B, 3, 20, 20)
            
        Returns:
            delta_action: 预测的动作增量 (B, action_dim)
        """
        # 提取特征
        features = self.extract_features(forces_l, forces_r)
        
        # MLP预测
        delta_action = self.mlp(features)
        
        return delta_action
    
    def predict(self, forces_l, forces_r):
        """预测模式，返回详细信息"""
        self.eval()
        with torch.no_grad():
            features = self.extract_features(forces_l, forces_r)
            delta_action = self.mlp(features)
            
            return {
                'delta_action': delta_action,
                'features': features,
                'feature_l': features[:, :self.feature_dim],
                'feature_r': features[:, self.feature_dim:]
            }


def compute_feature_mlp_losses(predictions, targets, config=None):
    """
    计算Feature-MLP损失
    
    Args:
        predictions: 模型预测的动作增量 (B, action_dim)
        targets: 真实动作增量 (B, action_dim)
        config: 损失配置
        
    Returns:
        loss: 总损失
        metrics: 损失分解字典
    """
    if config is None:
        config = {}
    
    # 主要损失：Huber Loss (对异常值更鲁棒)
    if config.get('loss_type', 'huber') == 'huber':
        delta = config.get('huber_delta', 1.0)
        main_loss = F.huber_loss(predictions, targets, delta=delta)
    elif config.get('loss_type') == 'mse':
        main_loss = F.mse_loss(predictions, targets)
    elif config.get('loss_type') == 'l1':
        main_loss = F.l1_loss(predictions, targets)
    else:
        main_loss = F.huber_loss(predictions, targets, delta=1.0)
    
    # 总损失
    total_loss = main_loss
    
    # 计算指标
    with torch.no_grad():
        mae = F.l1_loss(predictions, targets)
        mse = F.mse_loss(predictions, targets)
        
        # 计算各个轴的误差
        errors_per_axis = torch.abs(predictions - targets).mean(dim=0)
    
    metrics = {
        'main_loss': main_loss.item(),
        'total_loss': total_loss.item(),
        'mae': mae.item(),
        'mse': mse.item(),
        'rmse': torch.sqrt(mse).item(),
        'error_x': errors_per_axis[0].item() if len(errors_per_axis) > 0 else 0,
        'error_y': errors_per_axis[1].item() if len(errors_per_axis) > 1 else 0,
        'error_z': errors_per_axis[2].item() if len(errors_per_axis) > 2 else 0,
    }
    
    return total_loss, metrics


def create_feature_mlp(config):
    """
    根据配置创建Feature-MLP模型
    
    Args:
        config: 模型配置字典
        
    Returns:
        model: FeatureMLP模型实例
    """
    return FeatureMLP(
        feature_dim=config.get('feature_dim', 128),
        action_dim=config.get('action_dim', 3),
        hidden_dims=config.get('hidden_dims', [512, 512, 512]),
        dropout_rate=config.get('dropout_rate', 0.1),
        pretrained_encoder_path=config.get('pretrained_encoder_path', None)
    )


if __name__ == '__main__':
    # 测试模型
    config = {
        'feature_dim': 128,
        'action_dim': 3,
        'hidden_dims': [512, 512, 512],
        'dropout_rate': 0.1,
        'pretrained_encoder_path': '/path/to/pretrained/tactile_encoder.pt'  # 示例路径
    }
    
    model = create_feature_mlp(config)
    
    # 测试输入
    batch_size = 4
    forces_l = torch.randn(batch_size, 3, 20, 20)  # 左手触觉数据
    forces_r = torch.randn(batch_size, 3, 20, 20)  # 右手触觉数据
    target_actions = torch.randn(batch_size, 3)    # 目标动作
    
    # 前向传播
    print("=== 模型测试 ===")
    predicted_actions = model(forces_l, forces_r)
    
    print(f"输入形状:")
    print(f"  左手触觉: {forces_l.shape}")
    print(f"  右手触觉: {forces_r.shape}")
    print(f"输出形状:")
    print(f"  预测动作: {predicted_actions.shape}")
    
    # 计算损失
    loss_config = {'loss_type': 'huber', 'huber_delta': 1.0}
    loss, metrics = compute_feature_mlp_losses(predicted_actions, target_actions, loss_config)
    
    print(f"\n=== 损失测试 ===")
    print(f"总损失: {loss.item():.6f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    # 预测模式测试
    print(f"\n=== 预测模式测试 ===")
    results = model.predict(forces_l, forces_r)
    for key, value in results.items():
        print(f"  {key}: {value.shape}")
    
    print(f"\n=== 模型信息 ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"冻结参数数量: {total_params - trainable_params:,}")
