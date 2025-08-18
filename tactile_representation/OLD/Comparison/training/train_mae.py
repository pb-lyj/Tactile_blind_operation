"""
MAE训练脚本
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
from models.mae_model import create_tactile_mae, create_mae_loss, TactileMAEClassifier


class MAETrainer:
    """MAE训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config['output_dir'], f"mae_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # 初始化模型
        self.model = create_tactile_mae(config['model']).to(self.device)
        self.criterion = create_mae_loss(config['loss'])
        
        # 初始化优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # 学习率调度器（带warmup）
        self.warmup_epochs = config['training'].get('warmup_epochs', 10)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['training']['epochs'] - self.warmup_epochs
        )
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"设备: {self.device}")
        print(f"输出目录: {self.output_dir}")

    def warmup_lr(self, epoch):
        """学习率warmup"""
        if epoch < self.warmup_epochs:
            lr = self.config['training']['learning_rate'] * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        
        # Warmup学习率
        self.warmup_lr(epoch)
        
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['image'].to(self.device)
            
            # 前向传播
            outputs = self.model(inputs)
            loss_dict = self.criterion(inputs, outputs)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'Recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'Mask': f"{loss_dict['mask_ratio'].item():.3f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        return {
            'total_loss': total_loss / len(dataloader),
            'reconstruction_loss': total_recon_loss / len(dataloader)
        }

    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_recon_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                inputs = batch['image'].to(self.device)
                
                outputs = self.model(inputs)
                loss_dict = self.criterion(inputs, outputs)
                
                total_loss += loss_dict['total_loss'].item()
                total_recon_loss += loss_dict['reconstruction_loss'].item()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'reconstruction_loss': total_recon_loss / len(dataloader)
        }

    def train(self, train_loader, val_loader):
        """完整训练过程"""
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_metrics)
            
            # 验证
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics)
            
            # 更新学习率（在warmup之后）
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            # 打印结果
            print(f"Train Loss: {train_metrics['total_loss']:.4f}, "
                  f"Val Loss: {val_metrics['total_loss']:.4f}")
            
            # 保存最佳模型
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                self.save_model('best_model.pth')
            
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
            'recon_loss': [m['reconstruction_loss'] for m in self.train_losses]
        }
        return self.model, loss_history

    def train_single_dataset(self, dataloader):
        """单数据集训练过程 - 与Prototype Discovery保持一致"""
        best_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['patience']
        
        # 记录损失历史
        loss_fields = ['recon_loss']
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
                outputs = self.model(batch)
                loss_dict = self.criterion(batch, outputs)
                loss = loss_dict['total_loss']
                recon_loss = loss_dict['reconstruction_loss']
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                # 累积损失和指标
                batch_size = batch.size(0)
                total_loss += loss.item() * batch_size
                metrics_sum['recon_loss'] += recon_loss.item() * batch_size
            
            # 计算平均损失和指标
            avg_loss = total_loss / len(dataloader.dataset)
            avg_metrics = {k: v/len(dataloader.dataset) for k, v in metrics_sum.items()}
            
            # 学习率调度
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
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
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 总损失
        train_total = [m['total_loss'] for m in self.train_losses]
        val_total = [m['total_loss'] for m in self.val_losses]
        axes[0].plot(epochs, train_total, label='Train')
        axes[0].plot(epochs, val_total, label='Validation')
        axes[0].set_title('MAE Loss')
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
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()

    def visualize_reconstruction(self, dataloader, num_samples=8):
        """可视化MAE重建结果"""
        self.model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['image'][:num_samples].to(self.device)
                outputs = self.model(inputs)
                
                # 获取预测和遮罩
                pred = outputs['pred']
                mask = outputs['mask']
                
                # 将预测转换为图像
                pred_imgs = self.model.unpatchify(pred)
                
                # 创建遮罩图像
                mask_imgs = mask.unsqueeze(-1).repeat(1, 1, self.model.patch_size**2 * 3)
                mask_imgs = self.model.unpatchify(mask_imgs)
                
                # 转换为numpy数组
                inputs_np = inputs.cpu().numpy()
                pred_imgs_np = pred_imgs.cpu().numpy()
                mask_imgs_np = mask_imgs.cpu().numpy()
                
                # 绘制结果
                fig, axes = plt.subplots(4, num_samples, figsize=(2*num_samples, 8))
                
                for i in range(num_samples):
                    # 原始输入
                    input_mean = np.mean(inputs_np[i], axis=0)
                    axes[0, i].imshow(input_mean, cmap='viridis')
                    axes[0, i].set_title(f'Original' if i == 0 else '')
                    axes[0, i].axis('off')
                    
                    # 遮罩
                    mask_mean = np.mean(mask_imgs_np[i], axis=0)
                    axes[1, i].imshow(mask_mean, cmap='gray')
                    axes[1, i].set_title(f'Mask' if i == 0 else '')
                    axes[1, i].axis('off')
                    
                    # 重建结果
                    pred_mean = np.mean(pred_imgs_np[i], axis=0)
                    axes[2, i].imshow(pred_mean, cmap='viridis')
                    axes[2, i].set_title(f'Reconstructed' if i == 0 else '')
                    axes[2, i].axis('off')
                    
                    # 带遮罩的原图
                    masked_input = inputs_np[i] * (1 - mask_imgs_np[i])
                    masked_input_mean = np.mean(masked_input, axis=0)
                    axes[3, i].imshow(masked_input_mean, cmap='viridis')
                    axes[3, i].set_title(f'Masked Input' if i == 0 else '')
                    axes[3, i].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'mae_reconstruction_samples.png'))
                plt.close()
                break


def get_default_config():
    """获取默认配置"""
    return {
        'model': {
            'img_size': 20,
            'patch_size': 4,
            'in_channels': 3,
            'embed_dim': 192,
            'depth': 6,
            'num_heads': 3,
            'decoder_embed_dim': 96,
            'decoder_depth': 4,
            'decoder_num_heads': 3,
            'mlp_ratio': 4.0,
            'mask_ratio': 0.75
        },
        'loss': {
            'patch_size': 4,
            'in_channels': 3
        },
        'data': {
            'data_root': './data/data25.7_aligned',
            'categories': None,
            'batch_size': 64,
            'val_split': 0.2,
            'num_workers': 4
        },
        'training': {
            'epochs': 200,
            'learning_rate': 1e-4,
            'weight_decay': 0.05,
            'warmup_epochs': 10,
            'save_freq': 20
        },
        'output_dir': './outputs/mae'
    }


def main():
    parser = argparse.ArgumentParser(description='训练MAE模型')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_root', type=str, default='./data/data25.7_aligned', 
                        help='数据根目录')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='遮罩比例')
    
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
def main(config):
    """
    主训练函数 - 支持多数据集和单数据集训练
    """
    # 设置日志
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train_mae.log")
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print("MAE Tactile Representation Training")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print("=" * 60)
    
    # 创建数据集
    dataset = TactileRepresentationDataset(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'] or [
            "cir_lar", "cir_med", "cir_sma",
            "rect_lar", "rect_med", "rect_sma", 
            "tri_lar", "tri_med", "tri_sma"
        ],
        start_frame=0,
        exclude_test_folders=True
    )
    
    # 分割训练和验证集
    val_size = int(len(dataset) * config['data']['val_split'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # 创建训练器
    trainer = MAETrainer(config)
    
    # 开始训练
    model, loss_history = trainer.train(train_loader, val_loader)
    
    # 保存训练损失曲线 - 所有损失在一张图上
    plot_all_losses_single_plot(
        loss_history, 
        save_path=os.path.join(output_dir, "training_loss_curves.png"),
        title="MAE Training Loss"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(output_dir, "loss_history.npy"), loss_history)
    
    # 可视化重建结果
    trainer.visualize_reconstruction(val_loader)
    
    print("✅ MAE模型训练完成!")
    return model, loss_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE训练脚本')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--data_root', type=str, help='数据根目录')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--mask_ratio', type=float, help='掩码比例')
    
    args = parser.parse_args()
    
    # 默认配置
    config = {
        'data': {
            'data_root': '/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
            'categories': None,
            'batch_size': 64,
            'val_split': 0.2,
            'num_workers': 4
        },
        'model': {
            'image_size': (20, 20),
            'patch_size': 4,
            'embed_dim': 256,
            'encoder_depth': 6,
            'decoder_depth': 4,
            'encoder_heads': 8,
            'decoder_heads': 8,
            'masking_ratio': 0.75
        },
        'training': {
            'epochs': 200,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'save_freq': 25,
            'patience': 20
        },
        'output': {
            'output_dir': os.path.join("./cluster/comparison_library", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_mae")
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
    if args.mask_ratio:
        config['model']['masking_ratio'] = args.mask_ratio
    
    main(config)


if __name__ == '__main__':
    main()
