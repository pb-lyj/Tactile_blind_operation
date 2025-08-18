"""
VQGAN训练脚本
基于CNN编码解码器骨干网络的VQGAN模型训练
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
from models.vqgan_model import create_tactile_vqgan, compute_vqgan_losses


def train_vqgan(config):
    """
    训练VQGAN模型
    Args:
        config: 配置字典
    """
    # 创建输出目录
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train_vqgan.log")
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print("VQGAN Training")
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
    generator, discriminator = create_tactile_vqgan(config['model'])
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    
    print(f"生成器参数数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数数量: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # 优化器
    gen_optimizer = torch.optim.Adam(
        generator.parameters(), 
        lr=config['training']['lr'], 
        betas=(0.5, 0.9),
        weight_decay=config['training']['weight_decay']
    )
    
    disc_optimizer = torch.optim.Adam(
        discriminator.parameters(), 
        lr=config['training']['disc_lr'], 
        betas=(0.5, 0.9),
        weight_decay=config['training']['weight_decay']
    )
    
    # 学习率调度器
    gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer, mode='min', factor=0.5, patience=10
    )
    
    disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        disc_optimizer, mode='min', factor=0.5, patience=10
    )

    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    patience = config['training']['patience']
    global_step = 0
    
    # 记录损失历史
    loss_fields = ['recon_loss', 'perceptual_loss', 'vq_loss', 'gen_adv_loss', 'disc_loss', 'perplexity']
    loss_history = {'epoch': [], 'total_loss': []}
    for field in loss_fields:
        loss_history[field] = []

    for epoch in range(1, config['training']['epochs'] + 1):
        generator.train()
        discriminator.train()
        
        total_gen_loss = 0
        total_disc_loss = 0
        metrics_sum = {field: 0 for field in loss_fields}
        
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{config['training']['epochs']}"):
            inputs = batch['image'].cuda()
            global_step += 1
            
            # 训练生成器
            gen_optimizer.zero_grad()
            
            outputs = generator(inputs)
            gen_loss, disc_loss, metrics = compute_vqgan_losses(
                inputs, outputs, discriminator, config['loss'], global_step
            )
            
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            gen_optimizer.step()
            
            # 训练判别器（在开始对抗训练后）
            if global_step >= config['loss'].get('discriminator_start', 30000):
                disc_optimizer.zero_grad()
                
                # 重新计算判别器损失（避免梯度累积问题）
                with torch.no_grad():
                    outputs = generator(inputs)
                
                _, disc_loss, _ = compute_vqgan_losses(
                    inputs, outputs, discriminator, config['loss'], global_step
                )
                
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                disc_optimizer.step()
            
            # 累积损失和指标
            batch_size = inputs.size(0)
            total_gen_loss += gen_loss.item() * batch_size
            total_disc_loss += disc_loss.item() * batch_size
            for key in loss_fields:
                if key in metrics:
                    metrics_sum[key] += metrics[key] * batch_size
        
        # 计算平均损失和指标
        avg_gen_loss = total_gen_loss / len(dataset)
        avg_disc_loss = total_disc_loss / len(dataset)
        avg_metrics = {k: v/len(dataset) for k, v in metrics_sum.items()}
        
        # 学习率调度
        gen_scheduler.step(avg_gen_loss)
        disc_scheduler.step(avg_disc_loss)
        
        current_gen_lr = gen_optimizer.param_groups[0]['lr']
        current_disc_lr = disc_optimizer.param_groups[0]['lr']
        
        # 打印训练信息
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"  Generator Loss: {avg_gen_loss:.6f}")
        print(f"  Discriminator Loss: {avg_disc_loss:.6f}")
        print(f"  Generator LR: {current_gen_lr:.6e}")
        print(f"  Discriminator LR: {current_disc_lr:.6e}")
        print("  Individual Losses:")
        for key, value in avg_metrics.items():
            print(f"    {key}: {value:.6f}")
        print("-" * 50)
        
        # 记录损失历史
        loss_history['epoch'].append(epoch)
        loss_history['total_loss'].append(avg_gen_loss)
        for key, value in avg_metrics.items():
            if key != 'gen_total_loss':  # 避免重复添加total_loss
                loss_history[key].append(value)
        
        # 早停检查
        if avg_gen_loss < best_loss:
            best_loss = avg_gen_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                'disc_optimizer_state_dict': disc_optimizer.state_dict(),
                'epoch': epoch,
                'gen_loss': avg_gen_loss,
                'disc_loss': avg_disc_loss,
                'global_step': global_step,
                'config': config
            }, os.path.join(output_dir, "best_model.pt"))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停：{patience} 个epoch没有改善")
            break
        
        # 定期保存检查点
        if epoch % 10 == 0:
            visualize_reconstruction_and_quantization(generator, loader, output_dir, epoch)
    
    # 保存最终模型
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict(),
        'epoch': epoch,
        'gen_loss': avg_gen_loss,
        'disc_loss': avg_disc_loss,
        'global_step': global_step,
        'config': config
    }, os.path.join(output_dir, "final_model.pt"))
    
    # 保存训练损失曲线
    plot_all_losses_single_plot(
        loss_history, 
        save_path=os.path.join(output_dir, "training_loss_curves.png"),
        title="VQGAN Training Loss"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(output_dir, "loss_history.npy"), loss_history)
    
    # 最终可视化
    visualize_reconstruction_and_quantization(generator, loader, output_dir, "final")
    
    print("✅ VQGAN训练完成!")
    return generator, loss_history


def visualize_reconstruction_and_quantization(model, dataloader, output_dir, epoch, num_samples=8):
    """可视化重建结果和量化信息"""
    model.eval()
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'][:num_samples].cuda()
            outputs = model(inputs)
            reconstructions = outputs['reconstructed']
            
            # 保存重建图像
            save_physicalXYZ_images(
                inputs.cpu().numpy(),
                os.path.join(output_dir, f"original_samples_epoch_{epoch}.png")
            )
            
            save_physicalXYZ_images(
                reconstructions.cpu().numpy(),
                os.path.join(output_dir, f"reconstructed_samples_epoch_{epoch}.png")
            )
            
            # 保存量化信息
            perplexity = outputs['perplexity'].item()
            with open(os.path.join(output_dir, f"quantization_info_epoch_{epoch}.txt"), 'w') as f:
                f.write(f"Epoch: {epoch}\n")
                f.write(f"Perplexity: {perplexity:.4f}\n")
                f.write(f"Codebook Usage: {perplexity:.2f}/{model.quantizer.num_embeddings}\n")
            
            break
    
    model.train()


def main(config):
    """
    主训练函数
    """
    return train_vqgan(config)


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
            'in_channels': 3,
            'latent_dim': 256,
            'embedding_dim': 256,
            'num_embeddings': 1024,
            'commitment_cost': 0.25,
            'disc_ndf': 64
        },
        'loss': {
            'perceptual_weight': 0.1,
            'vq_weight': 1.0,
            'disc_weight': 0.1,
            'discriminator_start': 30000
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'lr': 1e-4,
            'disc_lr': 1e-4,
            'weight_decay': 1e-5,
            'patience': 20
        },
        'output': {
            'output_dir': os.path.join("./prototype_library", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_vqgan")
        }
    }
    
    main(config)
