"""
Feature-MLP模型 - 基于预训练触觉特征的行为克隆
使用预训练的CNN自编码器提取左右手触觉特征，然后用MLP学习动作映射
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# 获取项目根路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

try:
    from tactile_representation.Prototype_Discovery.models.cnn_autoencoder import TactileCNNAutoencoder
except ImportError:
    print("警告: 无法导入 TactileCNNAutoencoder")
    TactileCNNAutoencoder = None


class FeatureMLP(nn.Module):
    """
    Feature-MLP模型：基于预训练触觉特征的行为克隆
    
    架构：
    1. 预训练CNN编码器提取左右手触觉特征 (2 × 128维)
    2. 特征连接后输入MLP (256 → 256 → 128 → 3)
    3. 输出3维位置增量 (dx, dy, dz)
    """
    
    def __init__(self, 
                 feature_dim=128,           # 单手特征维度
                 action_dim=3,              # 输出动作维度 (dx, dy, dz)
                 hidden_dims=[256, 128],    # 隐藏层维度：256 → 128
                 dropout_rate=0.25,         # 提高Dropout到0.25
                 pretrained_encoder_path=None):
        """
        Args:
            feature_dim: 单手触觉特征维度
            action_dim: 输出动作维度
            hidden_dims: MLP隐藏层维度列表
            dropout_rate: Dropout比率
            pretrained_encoder_path: 预训练编码器权重路径
        """
        super(FeatureMLP, self).__init__()
        
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        
        # 加载预训练的触觉特征提取器
        if TactileCNNAutoencoder is not None:
            self.tactile_encoder = TactileCNNAutoencoder(
                in_channels=3, 
                latent_dim=feature_dim
            )
            
            # 加载预训练权重
            if pretrained_encoder_path is not None and os.path.exists(pretrained_encoder_path):
                print(f"加载预训练触觉编码器: {pretrained_encoder_path}")
                checkpoint = torch.load(pretrained_encoder_path, map_location='cpu')
                
                # 检查checkpoint格式，提取模型状态字典
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        # 标准训练checkpoint格式
                        model_state = checkpoint['model_state_dict']
                        print("📦 检测到训练checkpoint格式，提取model_state_dict")
                    elif 'state_dict' in checkpoint:
                        # 另一种常见格式
                        model_state = checkpoint['state_dict']
                        print("📦 检测到state_dict格式")
                    else:
                        # 直接的状态字典
                        model_state = checkpoint
                        print("📦 检测到直接状态字典格式")
                else:
                    model_state = checkpoint
                
                # 加载状态字典
                self.tactile_encoder.load_state_dict(model_state, strict=True)
                print("✅ 成功加载预训练权重")
                
                # 打印checkpoint信息
                if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
                    print(f"📊 预训练模型信息: epoch {checkpoint['epoch']}")
                    
            else:
                print("⚠️  预训练权重路径无效，使用随机初始化")
            
            # 冻结特征提取器参数
            for param in self.tactile_encoder.parameters():
                param.requires_grad = False
            print("🔒 特征提取器参数已冻结")
        else:
            print("❌ 无法导入CNN编码器，将使用随机特征")
            self.tactile_encoder = None
        
        # 构建MLP网络
        # 输入维度: 左右手特征连接 = feature_dim * 2
        input_dim = feature_dim * 2
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
        
        # 统计参数
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"� 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
        
    def _initialize_weights(self):
        """初始化MLP权重"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, forces_l, forces_r):
        """
        前向传播
        
        Args:
            forces_l: 左手触觉力数据 (B, 3, 20, 20)
            forces_r: 右手触觉力数据 (B, 3, 20, 20)
            
        Returns:
            actions: 预测的动作增量 (B, 3) - [dx, dy, dz]
        """
        batch_size = forces_l.size(0)
        
        if self.tactile_encoder is not None:
            # 使用预训练编码器提取特征
            with torch.no_grad():  # 编码器已冻结，不需要梯度
                features_l = self.tactile_encoder.encoder(forces_l)  # (B, feature_dim)
                features_r = self.tactile_encoder.encoder(forces_r)  # (B, feature_dim)
        else:
            # 如果没有编码器，使用简单的全局平均池化作为特征
            features_l = torch.mean(forces_l.view(batch_size, -1), dim=1, keepdim=True)
            features_r = torch.mean(forces_r.view(batch_size, -1), dim=1, keepdim=True)
            # 扩展到指定的特征维度
            features_l = features_l.repeat(1, self.feature_dim)
            features_r = features_r.repeat(1, self.feature_dim)
        
        # 连接左右手特征
        combined_features = torch.cat([features_l, features_r], dim=1)  # (B, feature_dim*2)
        
        # MLP预测动作
        actions = self.mlp(combined_features)  # (B, action_dim)
        
        return actions


def compute_feature_mlp_losses(predictions, targets, config=None):
    """
    计算Feature-MLP损失
    
    Args:
        predictions: 模型预测的动作增量 (B, 3)
        targets: 真实动作增量 (B, 3)
        config: 损失配置
        
    Returns:
        total_loss: 总损失
        metrics: 损失分解字典
    """
    if config is None:
        config = {}
    
    # 主要损失：Huber Loss (对异常值更鲁棒)
    loss_type = config.get('loss_type', 'huber')
    if loss_type == 'huber':
        delta = config.get('huber_delta', 1.0)
        main_loss = F.huber_loss(predictions, targets, delta=delta)
    elif loss_type == 'mse':
        main_loss = F.mse_loss(predictions, targets)
    elif loss_type == 'l1':
        main_loss = F.l1_loss(predictions, targets)
    else:
        main_loss = F.huber_loss(predictions, targets, delta=1.0)
    
    # 总损失
    total_loss = main_loss
    
    # 计算指标
    with torch.no_grad():
        l1_error = F.l1_loss(predictions, targets)    # 平均L1误差
        l2_error = F.mse_loss(predictions, targets)   # 平均L2误差(MSE)
        
        # 获取最后一组预测值和真实值供观察
        last_pred = predictions[-1].cpu().numpy() if len(predictions) > 0 else None
        last_target = targets[-1].cpu().numpy() if len(targets) > 0 else None
    
    metrics = {
        'main_loss': main_loss.item(),
        'total_loss': total_loss.item(),
        'l1_error': l1_error.item(),
        'l2_error': l2_error.item(),
        'rmse': torch.sqrt(l2_error).item(),
        'last_prediction': last_pred.tolist() if last_pred is not None else [],
        'last_target': last_target.tolist() if last_target is not None else [],
    }
    
    return total_loss, metrics


if __name__ == '__main__':
    # 测试模型创建和前向传播
    print("🧪 测试Feature-MLP模型...")
    
    # 创建模型
    model = FeatureMLP(
        feature_dim=128,
        action_dim=3,
        hidden_dims=[512, 512, 512],
        dropout_rate=0.1,
        pretrained_encoder_path=None  # 测试时不加载预训练权重
    )
    model.eval()
    
    # 创建测试数据
    batch_size = 4
    forces_l = torch.randn(batch_size, 3, 20, 20)
    forces_r = torch.randn(batch_size, 3, 20, 20)
    true_actions = torch.randn(batch_size, 3)
    
    print(f"输入形状: forces_l {forces_l.shape}, forces_r {forces_r.shape}")
    
    # 前向传播
    with torch.no_grad():
        predicted_actions = model(forces_l, forces_r)
    
    print(f"输出形状: {predicted_actions.shape}")
    print(f"预测动作: {predicted_actions[0].numpy()}")
    
    # 测试损失计算
    loss, metrics = compute_feature_mlp_losses(predicted_actions, true_actions)
    print(f"测试损失: {loss.item():.6f}")
    print(f"测试指标: L1={metrics['l1_error']:.6f}, RMSE={metrics['rmse']:.6f}")
    
    print("✅ Feature-MLP模型测试完成！")