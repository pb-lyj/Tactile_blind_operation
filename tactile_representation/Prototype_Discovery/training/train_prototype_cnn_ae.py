"""
Improved Prototype Autoencoder 训练脚本
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from utils.logging import Logger
from utils.visualization import save_physicalXYZ_images, plot_all_losses_single_plot
from utils.data_utils import save_sample_weights_and_analysis
from datasets.tactile_dataset import TactileForcesDataset
from models.prototype_cnn_ae import ImprovedForcePrototypeAE, compute_improved_losses


def train_prototype_cnn_ae(config):
    """
    训练改进版原型自编码器
    Args:
        config: 配置字典
    """
    # 创建数据集
    dataset = TactileForcesDataset(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        start_frame=config['data']['start_frame'],
        exclude_test_folders=config['data']['exclude_test_folders']
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )

    # 创建模型
    model = ImprovedForcePrototypeAE(
        num_prototypes=config['model']['num_prototypes'],
        input_shape=config['model']['input_shape']
    ).cuda()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    patience = config['training']['patience']
    
    # 记录损失历史
    loss_fields = ['recon_loss', 'diversity_loss', 'entropy_loss', 'sparsity_loss', 'gini_coeff']
    loss_history = {'epoch': [], 'total_loss': []}
    for field in loss_fields:
        loss_history[field] = []

    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        total_loss = 0
        metrics_sum = {field: 0 for field in loss_fields}
        
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
            inputs = batch['image'].cuda()
            
            # 前向传播
            recon, weights, protos = model(inputs)
            
            # 计算损失
            loss, metrics = compute_improved_losses(
                inputs, recon, weights, protos,
                diversity_lambda=config['loss']['diversity_lambda'],
                entropy_lambda=config['loss']['entropy_lambda'],
                sparsity_lambda=config['loss']['sparsity_lambda']
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累积损失和指标
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v * batch_size
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(dataset)
        avg_metrics = {k: v/len(dataset) for k, v in metrics_sum.items()}
        
        # 学习率调度
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练信息
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"  Total Loss: {avg_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.6e}")
        print("  Individual Losses:")
        for key, value in avg_metrics.items():
            print(f"    {key}: {value:.6f}")
        print("-" * 50)
        
        # 记录损失历史
        loss_history['epoch'].append(epoch)
        loss_history['total_loss'].append(avg_loss)
        for key, value in avg_metrics.items():
            if key != 'total_loss':  # 避免重复添加total_loss
                loss_history[key].append(value)
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 
                      os.path.join(config['output']['output_dir'], "best_model.pt"))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停：{patience} 个epoch没有改善")
            break

    return model, loss_history


def main(config):
    """
    主训练函数
    """
    # 创建输出目录
    os.makedirs(config['output']['output_dir'], exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(config['output']['output_dir'], "train_improved.log")
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print("Improved Prototype Autoencoder Training")
    print(f"Output Directory: {config['output']['output_dir']}")
    print(f"Prototypes: {config['model']['num_prototypes']}")
    print(f"Feature Dim: {config['model']['feature_dim']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['lr']}")
    print("=" * 60)
    
    # 训练模型
    model, loss_history = train_prototype_cnn_ae(config)
    
    # 保存模型和原型
    prototypes_np = model.prototypes.detach().cpu().numpy()
    np.save(os.path.join(config['output']['output_dir'], "prototypes.npy"), prototypes_np)
    save_physicalXYZ_images(prototypes_np, config['output']['output_dir'])
    torch.save(model.state_dict(), 
              os.path.join(config['output']['output_dir'], "final_model.pt"))

    # 绘制训练损失曲线 - 所有损失在一张图上
    plot_all_losses_single_plot(
        loss_history, 
        save_path=os.path.join(config['output']['output_dir'], "training_loss_curves.png"),
        title="Improved Prototype AE Training Loss"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(config['output']['output_dir'], "loss_history.npy"), loss_history)

    # 创建数据集用于样本权重分析
    dataset = TactileForcesDataset(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        start_frame=config['data']['start_frame'],
        exclude_test_folders=config['data']['exclude_test_folders']
    )
    
    # 保存样本权重和可视化
    save_sample_weights_and_analysis(model, dataset, output_dir=config['output']['output_dir'])
    
    print("✅ Improved模型训练完成!")
    return model, loss_history


if __name__ == '__main__':
    # 默认配置
    config = {
        'data': {
            'data_root': '/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
            'categories': [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ],
            'start_frame': 0,
            'exclude_test_folders': True,
            'num_workers': 8
        },
        'model': {
            'num_prototypes': 8,
            'input_shape': (3, 20, 20),
            'feature_dim': 128
        },
        'training': {
            'batch_size': 64,
            'epochs': 50,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'patience': 20
        },
        'loss': {
            'diversity_lambda': 0.1,
            'entropy_lambda': 10.0,
            'sparsity_lambda': 0.01,
            'gini_lambda': 0.05
        },
        'output': {
            'output_dir': os.path.join("./tactile_representation/prototype_library", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_improved")
        }
    }
    
    main(config)
