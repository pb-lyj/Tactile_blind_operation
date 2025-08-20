#!/usr/bin/env python3
"""
测试修改后的 policy_dataset.py 中的 is_train 逻辑
"""

import sys
import os
sys.path.append('/home/lyj/Program_python/Tactile_blind_operation')

from policy_learn.datasets.policy_dataset import PolicyDataset, create_train_test_datasets

def test_policy_dataset_is_train():
    data_root = "/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned"
    
    print("=== 测试单独创建训练集 ===")
    train_dataset = PolicyDataset(
        data_root=data_root,
        categories=["cir_lar", "cir_med"],
        start_frame=0,
        train_ratio=0.8,
        random_seed=42,
        is_train=True,
        use_end_states=True,
        use_forces=True,
        use_resultants=True
    )
    
    print(f"\n训练集样本数: {len(train_dataset)}")
    train_sample = train_dataset[0]
    print(f"训练集样本键: {list(train_sample.keys())}")
    print(f"Action 形状: {train_sample['action'].shape}")
    
    print("\n=== 测试单独创建测试集 ===")
    test_dataset = PolicyDataset(
        data_root=data_root,
        categories=["cir_lar", "cir_med"],
        start_frame=0,
        train_ratio=0.8,  # 同样的比例
        random_seed=42,   # 同样的随机种子
        is_train=False,   # 但是加载测试集
        use_end_states=True,
        use_forces=True,
        use_resultants=True
    )
    
    print(f"\n测试集样本数: {len(test_dataset)}")
    test_sample = test_dataset[0]
    print(f"测试集样本键: {list(test_sample.keys())}")
    
    print("\n=== 测试便捷函数 ===")
    train_ds, test_ds = create_train_test_datasets(
        data_root=data_root,
        categories=["cir_lar"],
        train_ratio=0.8,
        random_seed=42,
        use_end_states=False,  # 简化测试
        use_forces=False,
        use_resultants=False
    )
    
    print(f"便捷函数 - 训练集大小: {len(train_ds)}")
    print(f"便捷函数 - 测试集大小: {len(test_ds)}")
    
    # 验证数据划分一致性
    print(f"\n=== 验证数据划分一致性 ===")
    print(f"训练集 + 测试集总数: {len(train_ds) + len(test_ds)}")
    
    # 检查样本
    train_sample_simple = train_ds[0]
    test_sample_simple = test_ds[0]
    print(f"训练集样本键: {list(train_sample_simple.keys())}")
    print(f"测试集样本键: {list(test_sample_simple.keys())}")
    print(f"训练集 action 形状: {train_sample_simple['action'].shape}")
    print(f"测试集 action 形状: {test_sample_simple['action'].shape}")

if __name__ == "__main__":
    test_policy_dataset_is_train()
