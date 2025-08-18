"""
CNN自编码器验证脚本
加载训练好的CNN自编码器进行验证和测试
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# 添加 Prototype_Discovery 目录到路径
prototype_discovery_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if prototype_discovery_path not in sys.path:
    sys.path.insert(0, prototype_discovery_path)

from utils.logging import Logger
from utils.visualization import save_physicalXYZ_images, plot_all_losses_single_plot
from utils.data_utils import save_sample_weights_and_analysis
from datasets.tactile_dataset import TactileForcesDataset
from models.cnn_autoencoder import TactileCNNAutoencoder, compute_cnn_autoencoder_losses


def validate_cnn_autoencoder(config):
    """
    验证CNN自编码器
    Args:
        config: 配置字典（包含模型路径）
        函数中按 键 配置
    """
    # 创建输出目录
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "validate_cnn_autoencoder.log")
    sys.stdout = Logger(log_file)
    
    # 获取模型路径
    model_path = config['model']['model_path']
    
    print("=" * 60)
    print("CNN Autoencoder Validation")
    print(f"Output Directory: {output_dir}")
    print(f"Model Path: {model_path}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['validation']['batch_size']}")
    print("=" * 60)
    
    # 创建验证数据集（可以是测试集或验证集）
    dataset = TactileForcesDataset(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        start_frame=config['data']['start_frame'],
        exclude_test_folders=config['data']['exclude_test_folders'],
        normalize_method=config['data']['normalize_method'],
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=config['validation']['batch_size'], 
        shuffle=False,  # 验证时不需要shuffle
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )

    # 创建模型
    model = TactileCNNAutoencoder(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim']
    ).cuda()
    
    # 加载训练好的模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cuda')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 成功加载模型: {model_path}")
        print(f"模型训练轮次: {checkpoint.get('epoch', 'Unknown')}")
        print(f"模型训练损失: {checkpoint.get('loss', 'Unknown'):.6f}")
    else:
        print(f"❌ 模型文件不存在: {model_path}")
        return None, None
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 验证循环
    model.eval()
    total_loss = 0
    total_samples = 0
    loss_fields = ['recon_loss', 'l2_loss']
    metrics_sum = {field: 0 for field in loss_fields}
    
    # 存储验证结果
    all_inputs = []
    all_reconstructions = []
    all_latents = []
    all_losses = []
    
    print("开始验证...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating")):
            inputs = batch['image'].cuda()
            batch_size = inputs.size(0)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss, metrics = compute_cnn_autoencoder_losses(
                inputs, outputs, config['loss']
            )
            
            # 累积损失和指标
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            for key in loss_fields:
                metrics_sum[key] += metrics[key] * batch_size
            
            # 每10个batch保存部分结果用于可视化
            if batch_idx % 10 == 0 and batch_idx < config['validation']['save_samples_batches'] * 10:
                all_inputs.append(inputs.cpu().numpy())
                all_reconstructions.append(outputs['reconstructed'].cpu().numpy())
                all_latents.append(outputs['latent'].cpu().numpy())
                all_losses.append(loss.item())
    
    # 计算平均损失和指标
    avg_loss = total_loss / total_samples
    avg_metrics = {k: v/total_samples for k, v in metrics_sum.items()}
    
    # 打印验证结果
    print("\n" + "=" * 50)
    print("验证结果:")
    print(f"  总体损失: {avg_loss:.6f}")
    print("  各项损失:")
    for key, value in avg_metrics.items():
        print(f"    {key}: {value:.6f}")
    print("=" * 50)
    
    # 保存验证结果
    validation_results = {
        'avg_loss': avg_loss,
        'metrics': avg_metrics,
        'total_samples': total_samples,
        'model_path': config['model']['model_path']
    }
    
    # 保存验证结果到文件
    np.save(os.path.join(output_dir, "validation_results.npy"), validation_results)
    
    with open(os.path.join(output_dir, "validation_summary.txt"), 'w') as f:
        f.write(f"CNN Autoencoder Validation Results\n")
        f.write(f"Model Path: {config['model']['model_path']}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write(f"Average Loss: {avg_loss:.6f}\n")
        f.write("Individual Metrics:\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
    
    # 可视化验证结果
    if all_inputs:
        visualize_validation_results(
            all_inputs, all_reconstructions, all_latents, 
            output_dir, config['validation']['num_visualization_samples']
        )
    
    # 分析潜在空间
    if all_latents:
        analyze_latent_space(all_latents, output_dir)
    
    print("✅ CNN自编码器验证完成!")
    return validation_results, (all_inputs, all_reconstructions, all_latents)


def visualize_validation_results(inputs_list, reconstructions_list, latents_list, 
                                output_dir, num_samples=16):
    """可视化验证结果"""
    print("正在生成可视化结果...")
    
    # 创建reconstruction文件夹
    reconstruction_dir = os.path.join(output_dir, "reconstruction")
    os.makedirs(reconstruction_dir, exist_ok=True)
    
    # 合并所有批次的数据
    all_inputs = np.concatenate(inputs_list, axis=0)
    all_reconstructions = np.concatenate(reconstructions_list, axis=0)
    
    # 只取前num_samples个样本
    inputs_sample = all_inputs[:num_samples]
    reconstructions_sample = all_reconstructions[:num_samples]
    
    # 保存原始图像和重建图像到reconstruction文件夹
    save_physicalXYZ_images(
        inputs_sample,
        reconstruction_dir,
        prefix="validation_original"
    )
    
    save_physicalXYZ_images(
        reconstructions_sample,
        reconstruction_dir,
        prefix="validation_reconstructed"
    )
    
    # 计算重建误差可视化
    reconstruction_errors = np.abs(inputs_sample - reconstructions_sample)
    save_physicalXYZ_images(
        reconstruction_errors,
        reconstruction_dir,
        prefix="validation_errors"
    )
    
    print(f"✅ 可视化结果已保存到: {reconstruction_dir}")


def analyze_latent_space(latents_list, output_dir):
    """分析潜在空间"""
    print("正在分析潜在空间...")
    
    # 合并所有潜在向量
    all_latents = np.concatenate(latents_list, axis=0)
    
    # 计算统计信息
    latent_stats = {
        'mean': np.mean(all_latents, axis=0),
        'std': np.std(all_latents, axis=0),
        'min': np.min(all_latents, axis=0),
        'max': np.max(all_latents, axis=0)
    }
    
    # 保存潜在空间统计信息
    np.save(os.path.join(output_dir, "latent_space_stats.npy"), latent_stats)
    
    # 保存部分潜在向量样本
    sample_size = min(1000, all_latents.shape[0])
    latent_samples = all_latents[:sample_size]
    np.save(os.path.join(output_dir, "latent_samples.npy"), latent_samples)
    
    print(f"潜在空间维度: {all_latents.shape[1]}")
    print(f"潜在向量样本数: {all_latents.shape[0]}")
    print(f"潜在向量均值范围: [{latent_stats['mean'].min():.4f}, {latent_stats['mean'].max():.4f}]")
    print(f"潜在向量标准差范围: [{latent_stats['std'].min():.4f}, {latent_stats['std'].max():.4f}]")


def main(config):
    """
    主验证函数 - 不允许自动搜索，必须明确指定模型路径
    """
    if not config['model'].get('model_path'):
        print("❌ 错误：必须指定模型路径")
        print("请使用以下方式之一:")
        print("1. 在配置中设置 model_path")
        print("2. 使用命令行参数: --model_path /path/to/model.pt")
        print("3. 模型路径规则:")
        print("   - CNN AutoEncoder: ../prototype_library/YYYY-MM-DD_HH-MM-SS_cnn_autoencoder/best_model.pt")
        print("   - 或使用 final_model.pt")
        return None, None
    
    if not os.path.exists(config['model']['model_path']):
        print(f"❌ 错误：模型文件不存在: {config['model']['model_path']}")
        print("请检查路径是否正确")
        return None, None
    
    return validate_cnn_autoencoder(config)


if __name__ == '__main__':
    # 验证配置（与训练配置类似，但可能使用不同的数据集）
    config = {
        'data': {
            # 可以使用测试数据集或不同的验证数据集
            'data_root': '/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
            'categories': [
                "tri_lar", "tri_med", "tri_sma"
            ],
            'start_frame': 0,
            'exclude_test_folders': False,  # 验证时可能包含测试文件夹
            'num_workers': 8,
            'normalize_method': 'quantile_zscore'
        },
        'model': {
            'in_channels': 3,
            'latent_dim': 256,
            'model_path': None  # 设置为None，将自动寻找最新模型
        },
        'loss': {
            'l2_lambda': 0.001
        },
        'validation': {
            'batch_size': 32,  # 验证时可以用较小的batch size
            'save_samples_batches': 5,  # 保存多少个批次的结果用于可视化
            'num_visualization_samples': 16  # 可视化多少个样本
        },
        'output': {
            'output_dir': os.path.join("./validation_results", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_cnn_autoencoder")
        }
    }
    
    # 可以在这里指定特定的模型路径，例如:
    # config['model']['model_path'] = "./prototype_library/2025-08-17_10-30-00_cnn_autoencoder/best_model.pt"
    
    main(config)
