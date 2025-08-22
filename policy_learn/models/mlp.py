"""
MLP策略模型 - 低维策略网络
输入: resultant_force[6] + resultant_moment[6] = 12维
输出: delta_action_nextstep[3] = 3维
特性: 无时序下界；Huber/L2损失；隐藏层256×3；输入z-score标准化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TactilePolicyMLP(nn.Module):
    """
    触觉策略MLP模型 - 低维策略网络
    作为无时序下界基准模型
    """
    def __init__(self, input_dim=12, output_dim=3, hidden_dim=256, num_layers=3, 
                 dropout=0.1, use_normalization=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_normalization = use_normalization
        
        # 输入标准化 (z-score)
        if use_normalization:
            self.input_norm = nn.BatchNorm1d(input_dim)
        else:
            self.input_norm = nn.Identity()
        
        # 构建MLP网络
        layers = []
        
        # 第一层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        
        # 中间隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, resultant_force, resultant_moment):
        """
        前向传播
        
        Args:
            resultant_force: 合力张量 (B, 6) 或 (B, T, 6)
            resultant_moment: 合力矩张量 (B, 6) 或 (B, T, 6)
        
        Returns:
            delta_action: 动作增量 (B, 3) 或 (B, T, 3)
        """
        # 拼接输入特征
        x = torch.cat([resultant_force, resultant_moment], dim=-1)  # (B, 12) 或 (B, T, 12)
        
        # z-score标准化
        if self.use_normalization:
            if x.dim() == 3:  # (B, T, 12) - 序列数据
                B, T, D = x.shape
                x = x.view(B * T, D)
                x = self.input_norm(x)
                x = x.view(B, T, D)
            else:  # (B, 12) - 单帧数据
                x = self.input_norm(x)
        
        # 通过MLP网络
        delta_action = self.mlp(x)
        
        return delta_action


def compute_mlp_policy_losses(inputs, outputs, config):
    """
    计算MLP策略损失
    
    Args:
        inputs: 输入数据字典，包含 'target_delta_action'
        outputs: 模型输出张量
        config: 损失配置字典
    
    Returns:
        loss: 总损失
        metrics: 损失分解字典
    """
    predicted_delta = outputs
    target_delta = inputs['target_delta_action']
    
    # Huber损失（对异常值鲁棒）
    huber_loss = F.smooth_l1_loss(predicted_delta, target_delta)
    
    # L2损失
    l2_loss = F.mse_loss(predicted_delta, target_delta)
    
    # 加权总损失
    huber_weight = config.get('huber_weight', 1.0)
    l2_weight = config.get('l2_weight', 0.1)
    
    total_loss = huber_weight * huber_loss + l2_weight * l2_loss
    
    # 评估指标
    with torch.no_grad():
        mae = F.l1_loss(predicted_delta, target_delta)
        rmse = torch.sqrt(F.mse_loss(predicted_delta, target_delta))
    
    metrics = {
        'huber_loss': huber_loss.item(),
        'l2_loss': l2_loss.item(),
        'total_loss': total_loss.item(),
        'mae': mae.item(),
        'rmse': rmse.item()
    }
    
    return total_loss, metrics


def prepare_mlp_input_from_dataset(batch_data):
    """
    从数据集批次中准备MLP模型的输入
    
    Args:
        batch_data: 来自PolicyDataset或PolicyBatchedDataset的批次数据
    
    Returns:
        dict: MLP模型的输入字典
    """
    # 合并左右手的合力和合力矩
    resultant_force = torch.cat([
        batch_data['resultant_force_l'], 
        batch_data['resultant_force_r']
    ], dim=-1)  # (B, 6) 或 (B, T, 6)
    
    resultant_moment = torch.cat([
        batch_data['resultant_moment_l'], 
        batch_data['resultant_moment_r']
    ], dim=-1)  # (B, 6) 或 (B, T, 6)
    
    # 计算目标动作增量（从动作序列计算）
    actions = batch_data['action']  # (B, T, 6) 或 (B, 6)
    
    if actions.dim() == 3:  # 序列数据
        # 动作增量 = 下一步动作 - 当前动作
        current_actions = actions[:, :-1, :3]  # 前T-1步的位置 (B, T-1, 3)
        next_actions = actions[:, 1:, :3]  # 后T-1步的位置 (B, T-1, 3)
        target_delta_action = next_actions - current_actions  # (B, T-1, 3)
        
        # 对应调整输入特征
        resultant_force = resultant_force[:, :-1]  # (B, T-1, 6)
        resultant_moment = resultant_moment[:, :-1]  # (B, T-1, 6)
    else:  # 单帧数据
        target_delta_action = torch.zeros_like(actions[:, :3])  # (B, 3)
    
    return {
        'resultant_force': resultant_force,
        'resultant_moment': resultant_moment,
        'target_delta_action': target_delta_action
    }


# 便利函数
def create_tactile_policy_mlp(config):
    """创建触觉策略MLP模型"""
    return TactilePolicyMLP(
        input_dim=config.get('input_dim', 12),
        output_dim=config.get('output_dim', 3),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 3),
        dropout=config.get('dropout', 0.1),
        use_normalization=config.get('use_normalization', True)
    )


if __name__ == '__main__':
    # 简单测试
    config = {
        'input_dim': 12,
        'output_dim': 3,
        'hidden_dim': 256,
        'num_layers': 3,
        'dropout': 0.1,
        'use_normalization': True
    }
    
    model = create_tactile_policy_mlp(config)
    
    # 测试单帧输入
    resultant_force = torch.randn(4, 6)
    resultant_moment = torch.randn(4, 6)
    output = model(resultant_force, resultant_moment)
    
    print(f"输入合力形状: {resultant_force.shape}")
    print(f"输入合力矩形状: {resultant_moment.shape}")
    print(f"输出动作增量形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试损失计算
    inputs = {'target_delta_action': torch.randn_like(output)}
    loss_config = {'huber_weight': 1.0, 'l2_weight': 0.1}
    loss, metrics = compute_mlp_policy_losses(inputs, output, loss_config)
    
    print(f"总损失: {loss.item():.4f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")