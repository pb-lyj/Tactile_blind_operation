"""
Feature-MLP多步预测模型

基于预训练触觉编码器的多步动作序列预测
输入单帧触觉数据，输出H步位置增量序列
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# 添加项目路径以导入触觉编码器
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

try:
    from tactile_representation.Prototype_Discovery.models.cnn_autoencoder import TactileCNNAutoencoder
except ImportError:
    print("⚠️  无法导入TactileCNNAutoencoder，将使用简单替代")
    TactileCNNAutoencoder = None


class FeatureMLPMultiStep(nn.Module):
    """
    Feature-MLP多步预测模型：基于预训练触觉特征的多步行为预测
    
    架构：
    1. 预训练CNN编码器提取左右手触觉特征 (2 × 128维)
    2. 特征连接后输入MLP (256 → 256 → 128 → H*3)
    3. 输出重塑为H步位置增量序列 (B, H, 3)
    """
    
    def __init__(self, 
                 feature_dim=128,           # 单手特征维度
                 horizon=5,                 # 预测步数H
                 action_dim=3,              # 单步动作维度 (dx, dy, dz)
                 hidden_dims=[256, 128],    # 隐藏层维度
                 dropout_rate=0.25,         # Dropout率
                 pretrained_encoder_path=None):
        """
        Args:
            feature_dim: 单手触觉特征维度
            horizon: 预测的时间步数H
            action_dim: 单步动作维度
            hidden_dims: MLP隐藏层维度列表
            dropout_rate: Dropout比率
            pretrained_encoder_path: 预训练编码器权重路径
        """
        super(FeatureMLPMultiStep, self).__init__()
        
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # 初始化预训练触觉编码器
        if TactileCNNAutoencoder is not None:
            print("🔗 初始化预训练触觉编码器...")
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
        
        # 构建多步预测MLP网络
        # 输入维度: 左右手特征连接 = feature_dim * 2
        input_dim = feature_dim * 2
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层：预测H步动作序列
        output_dim = horizon * action_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 打印模型信息
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"🧠 模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    def forward(self, forces_l, forces_r):
        """
        前向传播
        
        Args:
            forces_l: 左手触觉力数据 (B, 3, 20, 20)
            forces_r: 右手触觉力数据 (B, 3, 20, 20)
            
        Returns:
            pred_seq: 预测的H步动作序列 (B, H, 3) - [[dx1,dy1,dz1], [dx2,dy2,dz2], ...]
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
        
        # MLP预测多步动作序列
        flat_predictions = self.mlp(combined_features)  # (B, H*3)
        
        # 重塑为序列形式
        pred_seq = flat_predictions.view(batch_size, self.horizon, self.action_dim)  # (B, H, 3)
        
        return pred_seq


def compute_multistep_losses(predictions, targets, config=None):
    """
    计算多步预测损失
    
    Args:
        predictions: 模型预测的H步动作序列 (B, H, 3)
        targets: 真实H步动作序列 (B, H, 3)
        config: 损失配置
        
    Returns:
        dict: 包含各种损失的字典
    """
    if config is None:
        config = {'loss_type': 'mse', 'step_weights': None}
    
    loss_type = config.get('loss_type', 'mse')
    step_weights = config.get('step_weights', None)
    
    batch_size, horizon, action_dim = predictions.shape
    
    # 逐步损失计算
    step_losses = []
    
    for h in range(horizon):
        pred_h = predictions[:, h, :]  # (B, 3)
        target_h = targets[:, h, :]    # (B, 3)
        
        if loss_type == 'mse':
            step_loss = F.mse_loss(pred_h, target_h, reduction='mean')
        elif loss_type == 'huber':
            delta = config.get('huber_delta', 1.0)
            step_loss = F.huber_loss(pred_h, target_h, delta=delta, reduction='mean')
        elif loss_type == 'l1':
            step_loss = F.l1_loss(pred_h, target_h, reduction='mean')
        else:
            step_loss = F.mse_loss(pred_h, target_h, reduction='mean')
        
        step_losses.append(step_loss)
    
    # 应用步长权重（如果提供）
    if step_weights is not None:
        assert len(step_weights) == horizon, f"权重长度{len(step_weights)}与horizon{horizon}不匹配"
        # 加权损失：每步损失乘以对应权重
        weighted_losses = [w * loss for w, loss in zip(step_weights, step_losses)]
        total_loss = sum(weighted_losses) / sum(step_weights)  # 标准化处理
    else:
        # 默认等权重
        total_loss = sum(step_losses) / horizon
    
    # 计算指标
    with torch.no_grad():
        # 总体MSE
        mse_loss = F.mse_loss(predictions, targets, reduction='mean')
        
        # 各步MSE
        step_mses = []
        for h in range(horizon):
            step_mse = F.mse_loss(predictions[:, h, :], targets[:, h, :], reduction='mean')
            step_mses.append(step_mse.item())
        
        # 最终步损失（通常最重要）
        final_step_loss = step_losses[-1].item()
        
        # 累积误差（每步误差累加）
        cumulative_error = 0
        for h in range(horizon):
            cumulative_error += F.l1_loss(predictions[:, h, :], targets[:, h, :], reduction='mean')
        cumulative_error = cumulative_error / horizon
    
    return {
        'total_loss': total_loss,
        'mse_loss': mse_loss.item(),
        'step_losses': [loss.item() for loss in step_losses],
        'step_mses': step_mses,
        'final_step_loss': final_step_loss,
        'cumulative_error': cumulative_error.item()
    }


def test_multistep_model():
    """测试多步预测模型"""
    print("🧪 测试Feature-MLP多步预测模型...")
    
    # 创建模型
    model = FeatureMLPMultiStep(
        feature_dim=128,
        horizon=5,
        action_dim=3,
        hidden_dims=[256, 128],
        dropout_rate=0.25,
        pretrained_encoder_path=None  # 测试时不加载预训练权重
    )
    
    # 模拟输入数据
    batch_size = 4
    forces_l = torch.randn(batch_size, 3, 20, 20)
    forces_r = torch.randn(batch_size, 3, 20, 20)
    
    print(f"\n📊 测试输入:")
    print(f"   左手触觉: {forces_l.shape}")
    print(f"   右手触觉: {forces_r.shape}")
    
    # 前向传播
    with torch.no_grad():
        pred_seq = model(forces_l, forces_r)
    
    print(f"\n📈 输出:")
    print(f"   预测序列: {pred_seq.shape}")
    print(f"   预测范围: [{pred_seq.min().item():.3f}, {pred_seq.max().item():.3f}]")
    
    # 测试损失计算
    target_seq = torch.randn_like(pred_seq)
    losses = compute_multistep_losses(pred_seq, target_seq)
    
    print(f"\n📉 损失测试:")
    print(f"   总损失: {losses['total_loss']:.4f}")
    print(f"   MSE损失: {losses['mse_loss']:.4f}")
    print(f"   最终步损失: {losses['final_step_loss']:.4f}")
    print(f"   各步损失: {[f'{x:.4f}' for x in losses['step_losses']]}")
    
    print("\n✅ 多步预测模型测试完成!")


if __name__ == '__main__':
    test_multistep_model()
