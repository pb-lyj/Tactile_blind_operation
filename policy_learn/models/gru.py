"""
GRU时序策略模型 - 低维策略（时序）
输入: resultant_force[6] + resultant_moment[6] {t}
输出: delta_action_nextstep[3]
特性: GRU隐层256；因果；动作平滑正则（Δu与jerk）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TactilePolicyGRU(nn.Module):
    """
    触觉策略GRU模型 - 低维时序策略网络
    """
    def __init__(self, input_dim=12, output_dim=3, hidden_dim=256, num_layers=2, 
                 dropout=0.1, use_normalization=True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_normalization = use_normalization
        
        # 输入标准化 
        if use_normalization:
            self.input_norm = nn.BatchNorm1d(input_dim)
        else:
            self.input_norm = nn.Identity()
        
        # GRU网络
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        for m in self.output_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, resultant_force, resultant_moment, hidden=None):
        """
        前向传播
        
        Args:
            resultant_force: 合力张量 (B, T, 6)
            resultant_moment: 合力矩张量 (B, T, 6)
            hidden: 隐藏状态 (num_layers, B, hidden_dim)
        
        Returns:
            delta_action: 动作增量 (B, T, 3)
            hidden: 最终隐藏状态
        """
        # 拼接输入特征
        x = torch.cat([resultant_force, resultant_moment], dim=-1)  # (B, T, 12)
        B, T, D = x.shape
        
        # z-score标准化
        if self.use_normalization:
            x = x.view(B * T, D)
            x = self.input_norm(x)
            x = x.view(B, T, D)
        
        # 通过GRU网络
        gru_out, hidden = self.gru(x, hidden)  # (B, T, hidden_dim)
        
        # 输出层
        delta_action = self.output_layer(gru_out)  # (B, T, 3)
        
        return delta_action, hidden
    
    def init_hidden(self, batch_size, device=None):
        """初始化隐藏状态"""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


def compute_gru_policy_losses(inputs, outputs, config):
    """
    计算GRU策略损失，包含动作平滑正则化
    
    Args:
        inputs: 输入数据字典，包含 'target_delta_action'
        outputs: 模型输出张量 (B, T, 3)
        config: 损失配置字典
    
    Returns:
        loss: 总损失
        metrics: 损失分解字典
    """
    predicted_delta = outputs  # (B, T, 3)
    target_delta = inputs['target_delta_action']  # (B, T, 3)
    
    # 主要损失：Huber损失
    huber_loss = F.smooth_l1_loss(predicted_delta, target_delta)
    
    # L2损失
    l2_loss = F.mse_loss(predicted_delta, target_delta)
    
    # 动作平滑正则化损失
    total_smoothing_loss = 0.0
    smoothing_metrics = {}
    
    # Δu损失：连续时间步的动作变化
    if config.get('delta_u_weight', 0.0) > 0:
        pred_diff = predicted_delta[:, 1:] - predicted_delta[:, :-1]  # (B, T-1, 3)
        target_diff = target_delta[:, 1:] - target_delta[:, :-1]  # (B, T-1, 3)
        delta_u_loss = F.mse_loss(pred_diff, target_diff)
        smoothing_metrics['delta_u_loss'] = delta_u_loss.item()
        total_smoothing_loss += config.get('delta_u_weight', 0.1) * delta_u_loss
    
    # Jerk损失：动作的二阶差分
    if config.get('jerk_weight', 0.0) > 0 and predicted_delta.size(1) >= 3:
        pred_diff2 = predicted_delta[:, 2:] - 2 * predicted_delta[:, 1:-1] + predicted_delta[:, :-2]
        target_diff2 = target_delta[:, 2:] - 2 * target_delta[:, 1:-1] + target_delta[:, :-2]
        jerk_loss = F.mse_loss(pred_diff2, target_diff2)
        smoothing_metrics['jerk_loss'] = jerk_loss.item()
        total_smoothing_loss += config.get('jerk_weight', 0.05) * jerk_loss
    
    # 加权总损失
    huber_weight = config.get('huber_weight', 1.0)
    l2_weight = config.get('l2_weight', 0.1)
    
    total_loss = huber_weight * huber_loss + l2_weight * l2_loss + total_smoothing_loss
    
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
    metrics.update(smoothing_metrics)
    
    return total_loss, metrics


def prepare_gru_input_from_dataset(batch_data):
    """
    从数据集批次中准备GRU模型的输入
    
    Args:
        batch_data: 来自PolicyBatchedDataset的批次数据
    
    Returns:
        dict: GRU模型的输入字典
    """
    # 合并左右手的合力和合力矩
    resultant_force = torch.cat([
        batch_data['resultant_force_l'], 
        batch_data['resultant_force_r']
    ], dim=-1)  # (B, T, 6)
    
    resultant_moment = torch.cat([
        batch_data['resultant_moment_l'], 
        batch_data['resultant_moment_r']
    ], dim=-1)  # (B, T, 6)
    
    # 计算目标动作增量
    actions = batch_data['action']  # (B, T, 6)
    current_actions = actions[:, :-1, :3]  # 前T-1步的位置 (B, T-1, 3)
    next_actions = actions[:, 1:, :3]  # 后T-1步的位置 (B, T-1, 3)
    target_delta_action = next_actions - current_actions  # (B, T-1, 3)
    
    # 对应调整输入特征（取前T-1步）
    resultant_force = resultant_force[:, :-1]  # (B, T-1, 6)
    resultant_moment = resultant_moment[:, :-1]  # (B, T-1, 6)
    
    return {
        'resultant_force': resultant_force,
        'resultant_moment': resultant_moment,
        'target_delta_action': target_delta_action
    }


# 便利函数
def create_tactile_policy_gru(config):
    """创建触觉策略GRU模型"""
    return TactilePolicyGRU(
        input_dim=config.get('input_dim', 12),
        output_dim=config.get('output_dim', 3),
        hidden_dim=config.get('hidden_dim', 256),
        num_layers=config.get('num_layers', 2),
        dropout=config.get('dropout', 0.1),
        use_normalization=config.get('use_normalization', True)
    )


if __name__ == '__main__':
    # 简单测试
    config = {
        'input_dim': 12,
        'output_dim': 3,
        'hidden_dim': 256,
        'num_layers': 2,
        'dropout': 0.1,
        'use_normalization': True
    }
    
    model = create_tactile_policy_gru(config)
    
    # 测试序列输入
    B, T = 4, 10
    resultant_force = torch.randn(B, T, 6)
    resultant_moment = torch.randn(B, T, 6)
    
    delta_action, hidden = model(resultant_force, resultant_moment)
    
    print(f"输入合力形状: {resultant_force.shape}")
    print(f"输入合力矩形状: {resultant_moment.shape}")
    print(f"输出动作增量形状: {delta_action.shape}")
    print(f"隐藏状态形状: {hidden.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试损失计算
    inputs = {'target_delta_action': torch.randn_like(delta_action)}
    loss_config = {
        'huber_weight': 1.0,
        'l2_weight': 0.1,
        'delta_u_weight': 0.1,
        'jerk_weight': 0.05
    }
    
    loss, metrics = compute_gru_policy_losses(inputs, delta_action, loss_config)
    print(f"总损失: {loss.item():.4f}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")