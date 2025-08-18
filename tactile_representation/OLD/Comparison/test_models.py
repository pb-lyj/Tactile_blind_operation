"""
测试脚本 - 验证模型和数据集是否正常工作
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tactile_representation.datasets.tactile_dataset import TactileRepresentationDataset
from tactile_representation.models.vqvae_model import create_tactile_vqvae
from tactile_representation.models.mae_model import create_tactile_mae
from tactile_representation.models.byol_model import create_tactile_byol


def test_dataset():
    """测试数据集加载"""
    print("=" * 50)
    print("测试数据集加载...")
    
    # 创建测试数据集（使用较少的数据）
    dataset = TactileRepresentationDataset(
        data_root="./data/data25.7_aligned",
        categories=["cir_lar"],  # 只使用一个类别进行测试
        task_type='reconstruction'
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试数据加载
    sample = dataset[0]
    print(f"样本shape: {sample['image'].shape}")
    print(f"数据类型: {sample['image'].dtype}")
    print(f"数据范围: [{sample['image'].min():.4f}, {sample['image'].max():.4f}]")
    
    # 可视化第一个样本
    if len(dataset) > 0:
        image = sample['image'].numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i in range(3):
            axes[i].imshow(image[i], cmap='viridis')
            axes[i].set_title(f'Channel {i}')
            axes[i].axis('off')
        
        plt.suptitle('Sample Tactile Data')
        plt.tight_layout()
        plt.savefig('./test_sample.png')
        plt.close()
        print("样本可视化已保存到 test_sample.png")
    
    return dataset


def test_vqvae_model():
    """测试VQ-VAE模型"""
    print("=" * 50)
    print("测试VQ-VAE模型...")
    
    model = create_tactile_vqvae()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 20, 20)
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
    
    print(f"输入shape: {test_input.shape}")
    print(f"重建shape: {output['reconstructed'].shape}")
    print(f"编码shape: {output['encoded'].shape}")
    print(f"量化shape: {output['quantized'].shape}")
    print(f"VQ损失: {output['vq_loss'].item():.4f}")
    
    # 测试编码和解码
    encoded = model.encode(test_input)
    decoded = model.decode(encoded)
    print(f"单独编码shape: {encoded.shape}")
    print(f"单独解码shape: {decoded.shape}")
    
    return model


def test_mae_model():
    """测试MAE模型"""
    print("=" * 50)
    print("测试MAE模型...")
    
    model = create_tactile_mae()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 20, 20)
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
    
    print(f"输入shape: {test_input.shape}")
    print(f"预测shape: {output['pred'].shape}")
    print(f"遮罩shape: {output['mask'].shape}")
    print(f"潜在表示shape: {output['latent'].shape}")
    print(f"遮罩比例: {output['mask'].float().mean().item():.3f}")
    
    # 测试编码
    encoded = model.encode(test_input)
    print(f"编码shape: {encoded.shape}")
    
    return model


def test_byol_model():
    """测试BYOL模型"""
    print("=" * 50)
    print("测试BYOL模型...")
    
    model = create_tactile_byol()
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建测试输入（两个增强版本）
    batch_size = 4
    test_input1 = torch.randn(batch_size, 3, 20, 20)
    test_input2 = torch.randn(batch_size, 3, 20, 20)
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input1, test_input2)
    
    print(f"输入1 shape: {test_input1.shape}")
    print(f"输入2 shape: {test_input2.shape}")
    print(f"在线预测1 shape: {output['online_pred_1'].shape}")
    print(f"在线预测2 shape: {output['online_pred_2'].shape}")
    print(f"目标投影1 shape: {output['target_proj_1'].shape}")
    print(f"目标投影2 shape: {output['target_proj_2'].shape}")
    
    # 测试编码
    encoded = model.encode(test_input1)
    print(f"编码shape: {encoded.shape}")
    
    # 测试目标网络更新
    model.update_target_network()
    print("目标网络更新成功")
    
    return model


def test_training_compatibility():
    """测试训练兼容性"""
    print("=" * 50)
    print("测试训练兼容性...")
    
    # 创建小数据集
    dataset = TactileRepresentationDataset(
        data_root="./data/data25.7_aligned",
        categories=["cir_lar"],
        task_type='reconstruction'
    )
    
    if len(dataset) == 0:
        print("警告: 没有找到数据，跳过训练兼容性测试")
        return
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 测试VQ-VAE训练一步
    vqvae = create_tactile_vqvae()
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=1e-4)
    
    batch = next(iter(dataloader))
    inputs = batch['image']
    
    optimizer.zero_grad()
    outputs = vqvae(inputs)
    
    # 简单的重建损失
    recon_loss = torch.nn.functional.mse_loss(outputs['reconstructed'], inputs)
    total_loss = recon_loss + outputs['vq_loss']
    
    total_loss.backward()
    optimizer.step()
    
    print(f"VQ-VAE训练步骤成功，损失: {total_loss.item():.4f}")
    
    print("所有测试通过！")


def main():
    """主测试函数"""
    print("开始测试触觉表征学习项目...")
    
    try:
        # 测试数据集
        dataset = test_dataset()
        
        # 测试模型
        vqvae = test_vqvae_model()
        mae = test_mae_model()
        byol = test_byol_model()
        
        # 测试训练兼容性
        test_training_compatibility()
        
        print("=" * 50)
        print("✅ 所有测试通过！项目设置正确。")
        print("\n可以开始训练:")
        print("python main_train.py vqvae --data_root ./data/data25.7_aligned")
        print("python main_train.py mae --data_root ./data/data25.7_aligned")
        print("python main_train.py byol --data_root ./data/data25.7_aligned")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
