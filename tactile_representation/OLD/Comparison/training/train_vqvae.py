"""
VQ-VAE训练脚本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from utils.logging import Logger
from utils.visualization import save_physicalXYZ_images, plot_all_losses_single_plot
from utils.data_utils import save_sample_weights_and_analysis
from datasets.tactile_dataset import TactileRepresentationDataset
from models.vqvae_model import create_tactile_vqvae, create_vqvae_loss, TactileVQVAEClassifier


class VQVAETrainer:
    """VQ-VAE训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config['output_dir'], f"vqvae_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # 初始化模型
        self.model = create_tactile_vqvae(config['model']).to(self.device)
        self.criterion = create_vqvae_loss(config['loss'])
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['training']['epochs']
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"设备: {self.device}")
        print(f"输出目录: {self.output_dir}")

    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['image'].to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            loss_dict = self.criterion(inputs, outputs)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_vq_loss += loss_dict['vq_loss'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'Recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'VQ': f"{loss_dict['vq_loss'].item():.4f}"
            })
        
        return {
            'total_loss': total_loss / len(dataloader),
            'reconstruction_loss': total_recon_loss / len(dataloader),
            'vq_loss': total_vq_loss / len(dataloader)
        }

    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                inputs = batch['image'].to(self.device)
                
                outputs = self.model(inputs)
                loss_dict = self.criterion(inputs, outputs)
                
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['reconstruction_loss'].item()
                total_vq_loss += loss_dict['vq_loss'].item()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'reconstruction_loss': total_recon_loss / len(dataloader),
            'vq_loss': total_vq_loss / len(dataloader)
        }

    def train(self, train_loader, val_loader):
        """完整训练过程"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics)
            
            # 验证
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics)
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印结果
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # 保存最佳模型
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            # 早停检查
            if patience_counter >= self.config['training']['patience']:
                print(f"早停：{self.config['training']['patience']} 个epoch没有改善")
                break
            
            # 定期保存
            if (epoch + 1) % self.config['training']['save_freq'] == 0:
                self.save_model(f'model_epoch_{epoch+1}.pth')
                self.plot_losses()
        
        # 保存最终模型
        self.save_model('final_model.pth')
        self.plot_losses()
        
        # 构建loss_history用于统一可视化（与Prototype Discovery格式一致）
        loss_history = {
            'epoch': list(range(1, len(self.train_losses) + 1)),
            'total_loss': [m['total_loss'] for m in self.train_losses],
            'recon_loss': [m['reconstruction_loss'] for m in self.train_losses],
            'vq_loss': [m['vq_loss'] for m in self.train_losses]
        }
        
        return self.model, loss_history

    def train_single_dataset(self, dataloader):
        """单数据集训练过程 - 与Prototype Discovery保持一致"""
        best_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['patience']
        
        # 记录损失历史
        loss_fields = ['recon_loss', 'vq_loss']
        loss_history = {'epoch': [], 'total_loss': []}
        for field in loss_fields:
            loss_history[field] = []

        for epoch in range(1, self.config['training']['epochs'] + 1):
            self.model.train()
            total_loss = 0
            metrics_sum = {field: 0 for field in loss_fields}
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{self.config['training']['epochs']}"):
                batch = batch.cuda()
                
                # 前向传播
                vq_loss, data_recon, perplexity = self.model(batch)
                recon_loss = nn.MSELoss()(data_recon, batch)
                loss = recon_loss + vq_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 累积损失和指标
                batch_size = batch.size(0)
                total_loss += loss.item() * batch_size
                metrics_sum['recon_loss'] += recon_loss.item() * batch_size
                metrics_sum['vq_loss'] += vq_loss.item() * batch_size
            
            # 计算平均损失和指标
            avg_loss = total_loss / len(dataloader.dataset)
            avg_metrics = {k: v/len(dataloader.dataset) for k, v in metrics_sum.items()}
            
            # 学习率调度
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(avg_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印训练信息
            print(f"Epoch {epoch} Loss: {avg_loss:.4f} LR: {current_lr:.2e}")
            print("Metrics:", " ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items()))
            
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
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"早停：{patience} 个epoch没有改善")
                break

        return self.model, loss_history

    def save_model(self, filename):
        """保存模型"""
        filepath = os.path.join(self.output_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }, filepath)
        print(f"模型已保存: {filepath}")

    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"模型已加载: {filepath}")

    def plot_losses(self):
        """绘制损失曲线"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 总损失
        train_total = [m['total_loss'] for m in self.train_losses]
        val_total = [m['total_loss'] for m in self.val_losses]
        axes[0].plot(epochs, train_total, label='Train')
        axes[0].plot(epochs, val_total, label='Validation')
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # 重建损失
        train_recon = [m['reconstruction_loss'] for m in self.train_losses]
        val_recon = [m['reconstruction_loss'] for m in self.val_losses]
        axes[1].plot(epochs, train_recon, label='Train')
        axes[1].plot(epochs, val_recon, label='Validation')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        # VQ损失
        train_vq = [m['vq_loss'] for m in self.train_losses]
        val_vq = [m['vq_loss'] for m in self.val_losses]
        axes[2].plot(epochs, train_vq, label='Train')
        axes[2].plot(epochs, val_vq, label='Validation')
        axes[2].set_title('VQ Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()

    def visualize_reconstruction(self, dataloader, num_samples=8):
        """可视化重建结果"""
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['image'][:num_samples].to(self.device)
                outputs = self.model(inputs)
                reconstructions = outputs['reconstructed']
                
                # 转换为numpy数组
                inputs_np = inputs.cpu().numpy()
                recons_np = reconstructions.cpu().numpy()
                
                # 绘制结果
                fig, axes = plt.subplots(3, num_samples, figsize=(2*num_samples, 6))
                
                for i in range(num_samples):
                    # 原始输入的三个通道
                    for c in range(3):
                        axes[c, i].imshow(inputs_np[i, c], cmap='viridis')
                        axes[c, i].set_title(f'Input Ch{c}' if i == 0 else '')
                        axes[c, i].axis('off')
                    
                    # 重建结果的均值可视化
                    recon_mean = np.mean(recons_np[i], axis=0)
                    axes[2, i].imshow(recon_mean, cmap='viridis')
                    axes[2, i].set_title(f'Recon' if i == 0 else '')
                    axes[2, i].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'reconstruction_samples.png'))
                plt.close()
                break


def get_default_config():
    """获取默认配置"""
    return {
        'model': {
            'in_channels': 3,
            'latent_dim': 64,
            'num_embeddings': 512,
            'commitment_cost': 0.25,
            'hidden_dim': 128
        },
        'loss': {
            'reconstruction_weight': 1.0,
            'vq_weight': 1.0,
            'perceptual_weight': 0.0
        },
        'data': {
            'data_root': './data/data25.7_aligned',
            'categories': None,  # None表示使用所有类别
            'batch_size': 32,
            'val_split': 0.2,
            'num_workers': 4
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'save_freq': 10
        },
        'output_dir': './outputs/vqvae'
    }


def main():
    parser = argparse.ArgumentParser(description='训练VQ-VAE模型')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_root', type=str, default='./data/data25.7_aligned', 
                        help='数据根目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = get_default_config()
    
    # 命令行参数覆盖配置
    if args.data_root:
        config['data']['data_root'] = args.data_root
    if args.epochs:
        config['training']['epochs'] = args.epochs
def main(config):
    """
    主训练函数
    """
    # 设置日志
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train_vqvae.log")
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print("VQ-VAE Tactile Representation Training")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print("=" * 60)
    
    # 创建数据集
    dataset = TactileRepresentationDataset(
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
    
    # 创建训练器
    trainer = VQVAETrainer(config)
    
    # 开始训练（单数据集模式）
    model, loss_history = trainer.train_single_dataset(loader)
    
    # 保存训练损失曲线 - 所有损失在一张图上
    plot_all_losses_single_plot(
        loss_history, 
        save_path=os.path.join(output_dir, "training_loss_curves.png"),
        title="VQ-VAE Training Loss"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(output_dir, "loss_history.npy"), loss_history)
    
    # 可视化重建结果
    trainer.visualize_reconstruction(loader)
    
    print("✅ VQ-VAE模型训练完成!")
    return model, loss_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VQ-VAE训练脚本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='学习率')
    
    args = parser.parse_args()
    
    # 默认配置
    config = {
        'data': {
            'data_root': '/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
            'categories': None,
            'batch_size': 32,
            'val_split': 0.2,
            'num_workers': 4
        },
        'model': {
            'in_channels': 3,
            'latent_dim': 64,
            'num_embeddings': 512,
            'commitment_cost': 0.25,
            'hidden_dim': 128
        },
        'loss': {
            'reconstruction_weight': 1.0,
            'vq_weight': 1.0,
            'perceptual_weight': 0.0
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'save_freq': 25,
            'patience': 20
        },
        'output': {
            'output_dir': os.path.join("./cluster/comparison_library", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_vqvae")
        }
    }
    
    # 从配置文件加载
    if args.config:
        with open(args.config, 'r') as f:
            loaded_config = json.load(f)
        config.update(loaded_config)
    
    # 命令行参数覆盖
    if args.data_root:
        config['data']['data_root'] = args.data_root
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    main(config)


if __name__ == '__main__':
    main()
