#!/usr/bin/env python3
"""
测试 Feature-MLP 模型的完整流程
验证模型创建、数据加载和训练逻辑
"""

import sys
import os
import torch

# 添加项目路径
sys.path.append('/home/lyj/Program_python/Tactile_blind_operation')

from policy_learn.models.feature_mlp import FeatureMLP, compute_feature_mlp_losses
from policy_learn.dataset_dataloader.policy_dataset import create_train_test_datasets


def test_feature_mlp_model():
    """测试 Feature-MLP 模型"""
    print("=== 测试 Feature-MLP 模型 ===")
    
    # 配置
    config = {
        'feature_dim': 128,
        'action_dim': 3,
        'hidden_dims': [512, 512, 512],
        'dropout_rate': 0.1,
        'pretrained_encoder_path': None  # 使用随机初始化
    }
    
    # 创建模型
    model = FeatureMLP(
        feature_dim=config['feature_dim'],
        action_dim=config['action_dim'],
        hidden_dims=config['hidden_dims'],
        dropout_rate=config['dropout_rate'],
        pretrained_encoder_path=config['pretrained_encoder_path']
    )
    
    # 测试输入
    batch_size = 4
    forces_l = torch.randn(batch_size, 3, 20, 20)
    forces_r = torch.randn(batch_size, 3, 20, 20)
    target_actions = torch.randn(batch_size, 3)
    
    # 前向传播
    predicted_actions = model(forces_l, forces_r)
    
    print(f"模型创建成功!")
    print(f"输入形状: 左手={forces_l.shape}, 右手={forces_r.shape}")
    print(f"输出形状: {predicted_actions.shape}")
    
    # 测试损失计算
    loss_config = {'loss_type': 'huber', 'huber_delta': 1.0}
    loss, metrics = compute_feature_mlp_losses(predicted_actions, target_actions, loss_config)
    
    print(f"损失计算成功: {loss.item():.6f}")
    print(f"指标: {metrics}")
    
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"冻结参数数量: {total_params - trainable_params:,}")


def test_data_loading():
    """测试数据加载"""
    print("\n=== 测试数据加载 ===")
    
    data_root = "/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned"
    
    try:
        # 创建数据集
        train_dataset, test_dataset = create_train_test_datasets(
            data_root=data_root,
            categories=["cir_lar", "cir_med"],
            train_ratio=0.8,
            random_seed=42,
            start_frame=0,
            use_end_states=False,
            use_forces=True,
            use_resultants=False
        )
        
        print(f"数据集创建成功!")
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        
        # 测试数据样本
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"样本键: {list(sample.keys())}")
            
            if 'forces_l' in sample and 'forces_r' in sample:
                print(f"左手触觉数据形状: {sample['forces_l'].shape}")
                print(f"右手触觉数据形状: {sample['forces_r'].shape}")
                print(f"动作数据形状: {sample['action'].shape}")
                return True
            else:
                print("警告: 样本中缺少触觉数据")
                return False
        else:
            print("警告: 训练集为空")
            return False
            
    except Exception as e:
        print(f"数据加载失败: {e}")
        return False


def test_training_step():
    """测试训练步骤"""
    print("\n=== 测试训练步骤 ===")
    
    # 创建模型
    model = FeatureMLP(
        feature_dim=128,
        action_dim=3,
        hidden_dims=[256, 256],  # 简化模型
        dropout_rate=0.1,
        pretrained_encoder_path=None
    )
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟训练数据
    batch_size = 2
    seq_len = 10
    
    # 模拟序列数据
    forces_l = torch.randn(batch_size, seq_len, 3, 20, 20)
    forces_r = torch.randn(batch_size, seq_len, 3, 20, 20)
    actions = torch.randn(batch_size, seq_len, 6)
    
    # 提取位置信息
    positions = actions[:, :, :3]  # (B, T, 3)
    
    # 准备训练数据
    all_forces_l = []
    all_forces_r = []
    all_deltas = []
    
    for t in range(seq_len - 1):
        curr_forces_l = forces_l[:, t]  # (B, 3, 20, 20)
        curr_forces_r = forces_r[:, t]  # (B, 3, 20, 20)
        
        curr_pos = positions[:, t]      # (B, 3)
        next_pos = positions[:, t + 1]  # (B, 3)
        delta = next_pos - curr_pos     # (B, 3)
        
        all_forces_l.append(curr_forces_l)
        all_forces_r.append(curr_forces_r)
        all_deltas.append(delta)
    
    # 合并数据
    forces_l_input = torch.cat(all_forces_l, dim=0)  # (B*(T-1), 3, 20, 20)
    forces_r_input = torch.cat(all_forces_r, dim=0)  # (B*(T-1), 3, 20, 20)
    delta_targets = torch.cat(all_deltas, dim=0)     # (B*(T-1), 3)
    
    # 训练步骤
    model.train()
    optimizer.zero_grad()
    
    predicted_deltas = model(forces_l_input, forces_r_input)
    
    loss_config = {'loss_type': 'huber', 'huber_delta': 1.0}
    loss, metrics = compute_feature_mlp_losses(predicted_deltas, delta_targets, loss_config)
    
    loss.backward()
    optimizer.step()
    
    print(f"训练步骤成功!")
    print(f"输入形状: 左手={forces_l_input.shape}, 右手={forces_r_input.shape}")
    print(f"目标形状: {delta_targets.shape}")
    print(f"预测形状: {predicted_deltas.shape}")
    print(f"损失: {loss.item():.6f}")
    print(f"MAE: {metrics.get('mae', 0):.6f}")


def main():
    """主测试函数"""
    print("开始测试 Feature-MLP 系统...")
    
    # 测试模型
    test_feature_mlp_model()
    
    # 测试数据加载
    data_success = test_data_loading()
    
    # 测试训练步骤
    test_training_step()
    
    print("\n=== 测试总结 ===")
    print("✅ 模型创建和前向传播")
    print("✅ 损失计算")
    print("✅ 训练步骤")
    if data_success:
        print("✅ 数据加载")
        print("\n🎉 所有测试通过! 系统准备就绪.")
        print("\n使用方法:")
        print("python train_feature_mlp.py --config configs/feature_mlp_config.json")
    else:
        print("❌ 数据加载失败")
        print("\n⚠️  请检查数据路径和格式")


if __name__ == '__main__':
    main()
