"""
BYOL训练脚本
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

from tactile_representation.Prototype_Discovery.dataset_dataloader.tactile_dataset import TactileForcesDataset
from tactile_representation.models.byol_model import create_tactile_byol, create_byol_loss, create_data_augmentation, TactileBYOLClassifier
from utils.logging import Logger
from utils.visualization import plot_loss_curves
from utils.data_utils import save_physicalXYZ_images, save_sample_weights_and_analysis


class BYOLDataset(torch.utils.data.Dataset):
    """包装数据集以应用BYOL增强"""
    
    def __init__(self, base_dataset, augmentation):
        self.base_dataset = base_dataset
        self.augmentation = augmentation
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        batch = self.base_dataset[idx]
        image = batch['image']
        
        # 应用BYOL增强
        aug1, aug2 = self.augmentation(image)
        
        return {
            'image1': aug1,
            'image2': aug2,
            'original': image
        }


class BYOLTrainer:
    """BYOL训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config['output_dir'], f"byol_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # 初始化模型
        self.model = create_tactile_byol(config['model']).to(self.device)
        self.criterion = create_byol_loss()
        
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
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            image1 = batch['image1'].to(self.device)
            image2 = batch['image2'].to(self.device)
            
            # 前向传播
            outputs = self.model(image1, image2)
            loss_dict = self.criterion(outputs)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss_dict['total_loss'].backward()
            self.optimizer.step()
            
            # 更新目标网络
            self.model.update_target_network()
            
            # 统计
            total_loss += loss_dict['total_loss'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        return {
            'total_loss': total_loss / len(dataloader),
            'byol_loss': total_loss / len(dataloader)
        }

    def validate(self, dataloader):
        """验证"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                image1 = batch['image1'].to(self.device)
                image2 = batch['image2'].to(self.device)
                
                outputs = self.model(image1, image2)
                loss_dict = self.criterion(outputs)
                
                total_loss += loss_dict['total_loss'].item()
        
        return {
            'total_loss': total_loss / len(dataloader),
            'byol_loss': total_loss / len(dataloader)
        }

    def train(self, train_loader, val_loader):
        """完整训练过程"""
        best_val_loss = float('inf')
        
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
                self.save_model('best_model.pth')
            
            # 定期保存
            if (epoch + 1) % self.config['training']['save_freq'] == 0:
                self.save_model(f'model_epoch_{epoch+1}.pth')
                self.plot_losses()
        
        # 保存最终模型
        self.save_model('final_model.pth')
        self.plot_losses()
        
        # 返回模型和损失历史
        loss_history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        return self.model, loss_history

    def train_single_dataset(self, data_dir, epochs=50, early_stopping_patience=5):
        """
        为单一数据集训练BYOL模型 - 兼容Prototype Discovery格式
        
        Args:
            data_dir: 数据目录路径
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
        
        Returns:
            loss_history: 与Prototype Discovery格式一致的损失历史
        """
        print(f"🚀 开始BYOL训练单一数据集: {data_dir}")
        print(f"训练轮数: {epochs}")
        
        # 创建数据集
        full_dataset = TactileForcesDataset(
            data_root=data_dir,
            categories=None,  # 使用所有可用类别
            start_frame=0,
            exclude_test_folders=True
        )
        
        # 分割训练和验证集 (80/20)
        val_size = int(len(full_dataset) * 0.2)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        print(f"训练样本数: {train_size}, 验证样本数: {val_size}")
        
        # 创建数据增强
        augmentation = create_data_augmentation()
        train_byol_dataset = BYOLDataset(train_dataset, augmentation)
        val_byol_dataset = BYOLDataset(val_dataset, augmentation)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_byol_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_byol_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 初始化损失历史 - 与Prototype Discovery格式一致
        loss_history = {
            'total_loss': [],
            'byol_loss': [],
            'recon_loss': []  # BYOL没有重建损失，但为保持一致性设为与byol_loss相同
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("\n" + "="*60)
        print("开始训练BYOL模型")
        print("="*60)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            epoch_losses = {'total_loss': 0, 'byol_loss': 0}
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                image1 = batch['image1'].to(self.device)
                image2 = batch['image2'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # 前向传播
                z1, z2, p1, p2 = self.model(image1, image2)
                loss = self.criterion(p1, z2) + self.criterion(p2, z1)
                
                # 反向传播
                loss.backward()
                self.optimizer.step()
                
                # 更新EMA权重
                self.model.update_target_network()
                
                # 记录损失
                epoch_losses['total_loss'] += loss.item()
                epoch_losses['byol_loss'] += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
            # 计算平均损失
            avg_train_loss = epoch_losses['total_loss'] / num_batches
            avg_byol_loss = epoch_losses['byol_loss'] / num_batches
            
            # 验证阶段
            self.model.eval()
            val_losses = {'total_loss': 0, 'byol_loss': 0}
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    image1 = batch['image1'].to(self.device)
                    image2 = batch['image2'].to(self.device)
                    
                    z1, z2, p1, p2 = self.model(image1, image2)
                    loss = self.criterion(p1, z2) + self.criterion(p2, z1)
                    
                    val_losses['total_loss'] += loss.item()
                    val_losses['byol_loss'] += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_losses['total_loss'] / val_batches
            avg_val_byol_loss = val_losses['byol_loss'] / val_batches
            
            # 记录损失历史
            loss_history['total_loss'].append(avg_train_loss)
            loss_history['byol_loss'].append(avg_byol_loss)
            loss_history['recon_loss'].append(avg_byol_loss)  # 与byol_loss相同
            
            # 更新学习率
            self.scheduler.step()
            
            # 打印训练进度
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # 早停检查
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                self.save_model('best_byol_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"\n⚠️ 早停触发！验证损失连续{early_stopping_patience}轮未改善")
                    break
        
        print(f"\n✅ BYOL训练完成！最佳验证损失: {best_val_loss:.4f}")
        
        return loss_history

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
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # BYOL损失
        train_total = [m['total_loss'] for m in self.train_losses]
        val_total = [m['total_loss'] for m in self.val_losses]
        ax.plot(epochs, train_total, label='Train')
        ax.plot(epochs, val_total, label='Validation')
        ax.set_title('BYOL Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'))
        plt.close()

    def visualize_features(self, dataloader, num_samples=8):
        """可视化学习到的特征"""
        self.model.eval()
        
        features = []
        images = []
        
        with torch.no_grad():
            for batch in dataloader:
                original = batch['original'][:num_samples].to(self.device)
                feature = self.model.encode(original)
                
                features.append(feature.cpu().numpy())
                images.append(original.cpu().numpy())
                
                if len(features) * num_samples >= 64:  # 收集足够的样本
                    break
        
        features = np.vstack(features)
        images = np.vstack(images)
        
        # 使用PCA降维可视化特征
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            
            # PCA降维
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features)
            
            # 绘制特征分布
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=range(len(features_2d)), cmap='viridis', alpha=0.7)
            plt.colorbar(scatter)
            plt.title('BYOL Feature Visualization (PCA)')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.savefig(os.path.join(self.output_dir, 'feature_visualization.png'))
            plt.close()
            
        except ImportError:
            print("sklearn未安装，跳过特征可视化")

    def create_augmented_datasets(self, base_train_dataset, base_val_dataset):
        """创建包含增强的数据集"""
        augmentation = create_data_augmentation(self.config.get('augmentation', {}))
        
        train_dataset = BYOLDataset(base_train_dataset, augmentation)
        val_dataset = BYOLDataset(base_val_dataset, augmentation)
        
        return train_dataset, val_dataset


def get_default_config():
    """获取默认配置"""
    return {
        'model': {
            'projection_dim': 128,
            'prediction_dim': 128,
            'hidden_dim': 256,
            'moving_average_decay': 0.99
        },
        'augmentation': {
            'noise_std': 0.1,
            'rotation_angle': 15,
            'flip_prob': 0.5
        },
        'data': {
            'data_root': './data/data25.7_aligned',
            'categories': None,
            'batch_size': 64,
            'val_split': 0.2,
            'num_workers': 4
        },
        'training': {
            'epochs': 300,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'save_freq': 25
        },
        'output_dir': './outputs/byol'
    }


def main(config):
    """
    主训练函数 - 支持多数据集和单数据集训练
    """
    # 设置日志
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train_byol.log")
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print("BYOL Tactile Representation Training")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print("=" * 60)
    
    # 创建数据集
    dataset = TactileForcesDataset(
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
    
    # 创建训练器
    trainer = BYOLTrainer(config)
    
    # 创建包含增强的数据集
    train_dataset_aug, val_dataset_aug = trainer.create_augmented_datasets(
        train_dataset, val_dataset
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset_aug,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset_aug,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # 开始训练
    model, loss_history = trainer.train(train_loader, val_loader)
    
    # 保存训练损失曲线 - 所有损失在一张图上
    from utils.visualization import plot_all_losses_single_plot
    plot_all_losses_single_plot(
        loss_history, 
        save_path=os.path.join(output_dir, "training_loss_curves.png"),
        title="BYOL Training Loss"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(output_dir, "loss_history.npy"), loss_history)
    
    # 可视化特征
    trainer.visualize_features(val_loader)
    
    print("✅ BYOL模型训练完成!")
    return model, loss_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BYOL训练脚本')
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
            'batch_size': 64,
            'val_split': 0.2,
            'num_workers': 4
        },
        'model': {
            'image_size': (20, 20),
            'backbone': 'resnet18',
            'projection_dim': 128,
            'hidden_dim': 512,
            'temperature': 0.2,
            'tau': 0.996
        },
        'training': {
            'epochs': 300,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'save_freq': 25,
            'patience': 50
        },
        'output': {
            'output_dir': os.path.join("./cluster/comparison_library", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_byol")
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
