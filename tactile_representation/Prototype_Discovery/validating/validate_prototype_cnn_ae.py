"""
Prototype CNN AutoEncoder验证脚本
加载训练好的Prototype CNN AutoEncoder进行验证和测试
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
from models.prototype_cnn_ae import ImprovedForcePrototypeAE, compute_improved_losses


def validate_prototype_cnn_ae(config):
    """
    验证Prototype CNN AutoEncoder
    Args:
        config: 配置字典
    """
    # 创建输出目录
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "validate_prototype_cnn_ae.log")
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print("Prototype CNN AutoEncoder Validation")
    print(f"Output Directory: {output_dir}")
    print(f"Model Path: {config['model']['model_path']}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['validation']['batch_size']}")
    print("=" * 60)
    
    # 创建数据集
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
        shuffle=False, 
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )

    # 创建模型
    model = ImprovedForcePrototypeAE(
        num_prototypes=config['model']['num_prototypes'],
        input_shape=config['model']['input_shape']
    ).cuda()
    
    # 加载训练好的模型
    checkpoint = torch.load(config['model']['model_path'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ 成功加载模型: {config['model']['model_path']}")
    print(f"模型训练轮次: {checkpoint.get('epoch', 'Unknown')}")
    print(f"模型训练损失: {checkpoint.get('loss', 'Unknown'):.6f}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 验证循环
    print("开始验证...")
    model.eval()
    total_loss = 0
    loss_fields = ['recon_loss', 'diversity_loss', 'entropy_loss', 'sparsity_loss']
    metrics_sum = {field: 0 for field in loss_fields}
    
    # 用于保存潜在空间分析
    all_weights = []
    all_prototypes = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Validating")):
            inputs = batch['image'].cuda()
            
            # 前向传播
            recon, weights, protos = model(inputs)
            
            # 计算损失
            loss, metrics = compute_improved_losses(
                inputs, recon, weights, protos, 
                **config['loss']
            )
            
            # 累积损失和指标
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            for key in loss_fields:
                metrics_sum[key] += metrics[key] * batch_size
            
            # 收集权重和原型信息用于分析
            all_weights.append(weights.cpu().numpy())
            if batch_idx == 0:  # 只保存一次原型
                all_prototypes = protos[0].cpu().numpy()  # (K, C, H, W)
            
            # 保存前几个批次的可视化结果
            if batch_idx < config['validation']['save_samples_batches']:
                batch_output_dir = os.path.join(output_dir, "reconstruction", f"batch_{batch_idx}")
                os.makedirs(batch_output_dir, exist_ok=True)
                
                # 保存原始图像
                save_physicalXYZ_images(
                    inputs[:config['validation']['num_visualization_samples']].cpu().numpy(),
                    batch_output_dir,
                    prefix="original"
                )
                
                # 保存重建图像
                save_physicalXYZ_images(
                    recon[:config['validation']['num_visualization_samples']].cpu().numpy(),
                    batch_output_dir,
                    prefix="reconstructed"
                )
                
                # 保存原型权重热图
                weights_viz = weights[:config['validation']['num_visualization_samples']].cpu().numpy()
                np.save(os.path.join(batch_output_dir, "prototype_weights.npy"), weights_viz)
    
    # 计算平均损失和指标
    avg_loss = total_loss / len(dataset)
    avg_metrics = {k: v/len(dataset) for k, v in metrics_sum.items()}
    
    print("\n" + "=" * 50)
    print("验证结果:")
    print(f"  总体损失: {avg_loss:.6f}")
    print("  各项损失:")
    for key, value in avg_metrics.items():
        print(f"    {key}: {value:.6f}")
    print("=" * 50)
    
    # 保存验证结果
    validation_results = {
        'total_loss': avg_loss,
        'metrics': avg_metrics,
        'config': config
    }
    
    # 可视化
    print("正在生成可视化结果...")
    reconstruction_dir = os.path.join(output_dir, "reconstruction")
    print(f"✅ 可视化结果已保存到: {reconstruction_dir}")
    
    # 原型权重分析
    print("正在分析原型权重分布...")
    all_weights = np.concatenate(all_weights, axis=0)  # (N, K)
    
    print(f"原型权重维度: {all_weights.shape}")
    print(f"权重均值范围: [{np.mean(all_weights, axis=0).min():.4f}, {np.mean(all_weights, axis=0).max():.4f}]")
    print(f"权重标准差范围: [{np.std(all_weights, axis=0).min():.4f}, {np.std(all_weights, axis=0).max():.4f}]")
    
    # 保存原型权重分析
    np.save(os.path.join(output_dir, "prototype_weights_analysis.npy"), all_weights)
    
    # 保存原型库
    if len(all_prototypes) > 0:
        save_physicalXYZ_images(
            all_prototypes,
            output_dir,
            prefix="prototype"
        )
        print(f"✅ 原型库已保存: {all_prototypes.shape}")
    
    # 保存验证结果到文件
    results_file = os.path.join(output_dir, "validation_results.txt")
    with open(results_file, 'w') as f:
        f.write("Prototype CNN AutoEncoder Validation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total Loss: {avg_loss:.6f}\n")
        f.write("Individual Losses:\n")
        for key, value in avg_metrics.items():
            f.write(f"  {key}: {value:.6f}\n")
        f.write("\n")
        f.write(f"Dataset Size: {len(dataset)}\n")
        f.write(f"Prototype Weights Shape: {all_weights.shape}\n")
        f.write(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    print("✅ Prototype CNN AutoEncoder验证完成!")
    return model, validation_results


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
        print("   - Prototype CNN: ../prototype_library/YYYY-MM-DD_HH-MM-SS_prototype_cnn/best_model.pt")
        print("   - 或使用 final_model.pt")
        return None, None
    
    if not os.path.exists(config['model']['model_path']):
        print(f"❌ 错误：模型文件不存在: {config['model']['model_path']}")
        print("请检查路径是否正确")
        return None, None
    
    return validate_prototype_cnn_ae(config)


if __name__ == '__main__':
    # 验证配置
    config = {
        'data': {
            'data_root': '/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
            'categories': [
                "tri_lar", "tri_med", "tri_sma"
            ],
            'start_frame': 0,
            'exclude_test_folders': False,
            'num_workers': 8,
            'normalize_method': 'quantile_zscore'
        },
        'model': {
            'num_prototypes': 8,
            'input_shape': [3, 20, 20],
            'model_path': None  # 设置为None，将自动寻找最新模型
        },
        'loss': {
            'diversity_lambda': 1.0,
            'entropy_lambda': 0.1,
            'sparsity_lambda': 0.01
        },
        'validation': {
            'batch_size': 32,
            'save_samples_batches': 5,
            'num_visualization_samples': 16
        },
        'output': {
            'output_dir': os.path.join("./validation_results", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_prototype_cnn_ae")
        }
    }
    
    main(config)
