"""
CNN自编码器训练脚本
使用CNN编码解码器进行触觉力数据重建训练
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
from dataset_dataloader.tactile_dataset import TactileForcesDataset
from models.cnn_autoencoder import TactileCNNAutoencoder, compute_cnn_autoencoder_losses


def train_cnn_autoencoder(config):
    """
    训练CNN自编码器
    Args:
        config: 配置字典
    """
    # 创建输出目录
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train_cnn_autoencoder.log")
    sys.stdout = Logger(log_file)
    reconstruction_dir = os.path.join(output_dir, "reconstruction")
    os.makedirs(reconstruction_dir, exist_ok=True)
    
    print("=" * 60)
    print("CNN Autoencoder Training")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['lr']}")
    print("=" * 60)
    
    # 创建数据集
    dataset = TactileForcesDataset(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        start_frame=config['data']['start_frame'],
        normalize_method=config['data']['normalize_method'],
        train_ratio=config['data'].get('train_ratio', 0.9),
        random_seed=config['data'].get('random_seed', 42),
        is_train=True
    )
    loader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )

    # 创建模型
    model = TactileCNNAutoencoder(
        in_channels=config['model']['in_channels'],
        latent_dim=config['model']['latent_dim']
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
    loss_fields = ['recon_loss', 'l2_loss']
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
            outputs = model(inputs)
            
            # 计算损失
            loss, metrics = compute_cnn_autoencoder_losses(
                inputs, outputs, config['loss']
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累积损失和指标
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            for key in loss_fields:
                metrics_sum[key] += metrics[key] * batch_size
        
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
        print("Metrics:", " ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items()))
        
        # 记录损失历史
        loss_history['epoch'].append(epoch)
        loss_history['total_loss'].append(avg_loss)
        for key, value in avg_metrics.items():
            if key != 'total_loss':  # 避免重复添加total_loss
                loss_history[key].append(value)
        
        # 每10个epoch可视化重建结果
        if epoch % 10 == 0:
            epoch_dir = os.path.join(reconstruction_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            print(f"正在生成第{epoch}轮的重建可视化...")
            visualize_reconstruction(model, loader, epoch_dir)
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': avg_loss,
                'config': config
            }, os.path.join(output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停：{patience} 个epoch没有改善")
            break
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': avg_loss,
        'config': config
    }, os.path.join(output_dir, "final_model.pt"))
    
    # 保存训练损失曲线
    plot_all_losses_single_plot(
        loss_history, 
        save_path=os.path.join(output_dir, "training_loss_curves.png"),
        title="CNN Autoencoder Training Loss"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(output_dir, "loss_history.npy"), loss_history)
    
    # 最终可视化重建结果
    print("正在生成最终的重建可视化...")
    visualize_reconstruction(model, loader, output_dir)
    
    print("✅ CNN自编码器训练完成!")
    return model, loss_history


def visualize_reconstruction(model, dataloader, output_dir, max_samples=4, batch_interval=10):
    """可视化重建结果"""
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 每10个batch保存一次
            if batch_idx % batch_interval == 0:
                inputs = batch['image'][:max_samples].cuda()
                outputs = model(inputs)
                reconstructions = outputs['reconstructed']
                
                # 根据batch_idx设置文件名
                prefix_original = f"batch_{batch_idx}_original"
                prefix_reconstructed = f"batch_{batch_idx}_reconstructed"
                
                # 保存原始图像和重建图像到指定的输出目录
                save_physicalXYZ_images(
                    inputs.cpu().numpy(),
                    output_dir,
                    prefix=prefix_original
                )
                
                save_physicalXYZ_images(
                    reconstructions.cpu().numpy(),
                    output_dir,
                    prefix=prefix_reconstructed
                )
                
                # print(f"已保存第{batch_idx}批次的重建可视化到 {output_dir}")
            
            # 限制最大保存批次数，避免生成过多文件
            if batch_idx >= 80:  # 最多保存前个batch中的每10个batch
                break


def main(config):
    """
    主训练函数
    """
    return train_cnn_autoencoder(config)


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
            'num_workers': 8,
            'normalize_method': 'minmax_255'
        },
        'model': {
            'in_channels': 3,
            'latent_dim': 128
        },
        'loss': {
            'l2_lambda': 0.001
        },
        'training': {
            'batch_size': 64,
            'epochs': 50,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'patience': 20
        },
        'output': {
            'output_dir': os.path.join("./prototype_library", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_cnn_autoencoder")
        }
    }
    
    main(config)
