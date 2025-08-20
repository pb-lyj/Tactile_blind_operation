#!/usr/bin/env python3
"""
测试修改后的 tactile_dataset.py
"""

import sys
import os
sys.path.append('/home/lyj/Program_python/Tactile_blind_operation')

from tactile_representation.Prototype_Discovery.dataset_dataloader.tactile_dataset import TactileForcesDataset, create_train_test_tactile_datasets

def test_tactile_dataset():
    # 设置数据根目录
    data_root = "/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned"
    
    print("=== 测试单独创建训练集 ===")
    train_dataset = TactileForcesDataset(
        data_root=data_root,
        categories=["cir_lar", "cir_med"],
        start_frame=0,
        train_ratio=0.8,
        random_seed=42,
        augment=False,
        normalize_method='quantile_zscore'
    )
    
    print(f"\n训练集样本数: {len(train_dataset)}")
    sample = train_dataset[0]
    print(f"样本形状: {sample['image'].shape}")
    print(f"样本数据范围: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")
    
    print("\n=== 测试单独创建测试集 ===")
    test_dataset = TactileForcesDataset(
        data_root=data_root,
        categories=["cir_lar", "cir_med"],
        start_frame=0,
        train_ratio=0.2,  # 加载测试集
        random_seed=42,
        augment=False,
        normalize_method='quantile_zscore'
    )
    
    print(f"\n测试集样本数: {len(test_dataset)}")
    
    print("\n=== 测试便捷函数 ===")
    train_ds, test_ds = create_train_test_tactile_datasets(
        data_root=data_root,
        categories=["cir_lar"],
        train_ratio=0.8,
        random_seed=42,
        augment_train=True,
        augment_test=False,
        normalize_method='minmax_255'
    )
    
    print(f"便捷函数 - 训练集大小: {len(train_ds)}")
    print(f"便捷函数 - 测试集大小: {len(test_ds)}")
    
    # 验证数据增强效果
    train_sample = train_ds[0]
    print(f"训练集样本形状: {train_sample['image'].shape}")
    print(f"训练集数据范围: [{train_sample['image'].min():.4f}, {train_sample['image'].max():.4f}]")

if __name__ == "__main__":
    test_tactile_dataset()
