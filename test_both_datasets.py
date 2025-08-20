#!/usr/bin/env python3
"""
测试 policy_dataset.py 和 tactile_dataset.py 的兼容性
"""

import sys
import os
sys.path.append('/home/lyj/Program_python/Tactile_blind_operation')

from tactile_representation.Prototype_Discovery.dataset_dataloader.policy_dataset import PolicyDataset, create_train_test_datasets
from tactile_representation.Prototype_Discovery.dataset_dataloader.tactile_dataset import TactileForcesDataset, create_train_test_tactile_datasets

def test_both_datasets():
    data_root = "/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned"
    
    print("=== 测试 Policy Dataset ===")
    policy_train, policy_test = create_train_test_datasets(
        data_root=data_root,
        categories=["cir_lar"],
        train_ratio=0.8,
        random_seed=42,
        chronology=True,
        load_end_states=False
    )
    print(f"Policy 训练集大小: {len(policy_train)}")
    print(f"Policy 测试集大小: {len(policy_test)}")
    
    policy_sample = policy_train[0]
    print(f"Policy 样本键: {list(policy_sample.keys())}")
    print(f"Action 形状: {policy_sample['action'].shape}")
    print(f"Forces 形状: {policy_sample['forces'].shape}")
    
    print("\n=== 测试 Tactile Dataset ===")
    tactile_train, tactile_test = create_train_test_tactile_datasets(
        data_root=data_root,
        categories=["cir_lar"],
        train_ratio=0.8,
        random_seed=42,
        augment_train=False,
        augment_test=False,
        normalize_method='minmax_255'
    )
    print(f"Tactile 训练集大小: {len(tactile_train)}")
    print(f"Tactile 测试集大小: {len(tactile_test)}")
    
    tactile_sample = tactile_train[0]
    print(f"Tactile 样本键: {list(tactile_sample.keys())}")
    print(f"Image 形状: {tactile_sample['image'].shape}")
    
    print("\n=== 验证随机种子一致性 ===")
    # 使用相同随机种子创建数据集
    policy_train_2, _ = create_train_test_datasets(
        data_root=data_root,
        categories=["cir_lar"],
        train_ratio=0.8,
        random_seed=42,
        chronology=True
    )
    
    tactile_train_2, _ = create_train_test_tactile_datasets(
        data_root=data_root,
        categories=["cir_lar"],
        train_ratio=0.8,
        random_seed=42,
        augment_train=False,
        normalize_method='minmax_255'
    )
    
    print(f"Policy 数据集大小一致性: {len(policy_train) == len(policy_train_2)}")
    print(f"Tactile 数据集大小一致性: {len(tactile_train) == len(tactile_train_2)}")
    print("(注意: 由于数据集结构不同，大小可能不同，但应该是确定性的)")

if __name__ == "__main__":
    test_both_datasets()
