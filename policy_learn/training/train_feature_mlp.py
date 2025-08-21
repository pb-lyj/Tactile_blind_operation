"""
训练 Feature-MLP 模型的脚本
基于预训练触觉特征进行行为克隆
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from utils.logging import Logger
from utils.visualization import plot_all_losses_single_plot
from policy_learn.models.feature_mlp import FeatureMLP, compute_feature_mlp_losses
from policy_learn.dataset_dataloader.policy_dataset import create_train_test_datasets


def feature_mlp_collate_fn(batch):
    """
    Feature-MLP专用collate函数
    将序列数据展开为单个时间步样本，用于单步预测
    """
    all_forces_l = []
    all_forces_r = []
    all_deltas = []
    all_categories = []
    
    for item in batch:
        forces_l = item['forces_l']  # (T, 3, 20, 20)
        forces_r = item['forces_r']  # (T, 3, 20, 20)
        actions = item['action']     # (T, 6) - 已经是增量数据
        category = item['category']
        
        # 只使用位置增量信息（前3维）
        deltas = actions[:, :3]      # (T, 3) - 直接使用增量数据
        seq_len = deltas.shape[0]
        
        # 将每个时间步展开为独立样本
        for t in range(seq_len):
            # 当前帧的触觉数据
            curr_forces_l = forces_l[t]  # (3, 20, 20)
            curr_forces_r = forces_r[t]  # (3, 20, 20)
            
            # 直接使用action文件中的增量数据
            delta = deltas[t]            # (3,)
            
            all_forces_l.append(curr_forces_l)
            all_forces_r.append(curr_forces_r)
            all_deltas.append(delta)
            all_categories.append(category)
    
    if len(all_forces_l) == 0:
        # 返回空batch
        return {
            'forces_l': torch.empty(0, 3, 20, 20),
            'forces_r': torch.empty(0, 3, 20, 20),
            'delta': torch.empty(0, 3),
            'category': []
        }
    
    # 转换为tensor
    return {
        'forces_l': torch.stack(all_forces_l),  # (N, 3, 20, 20)
        'forces_r': torch.stack(all_forces_r),  # (N, 3, 20, 20)
        'delta': torch.stack(all_deltas),       # (N, 3)
        'category': all_categories
    }


def collate_fn(batch):
    """
    自定义collate函数，处理变长序列，保持完整动作序列
    """
    # 获取batch中的最大序列长度
    max_len = max([item['action'].shape[0] for item in batch])
    
    batch_data = {
        'forces_l': [],
        'forces_r': [],
        'action': [],
        'category': [],
        'timestamps': [],
        'seq_lengths': []  # 记录原始序列长度
    }
    
    for item in batch:
        seq_len = item['action'].shape[0]
        batch_data['seq_lengths'].append(seq_len)
        
        # 对forces和action进行padding到最大长度
        forces_l_padded = torch.zeros(max_len, *item['forces_l'].shape[1:])
        forces_r_padded = torch.zeros(max_len, *item['forces_r'].shape[1:])
        action_padded = torch.zeros(max_len, *item['action'].shape[1:])
        timestamps_padded = torch.zeros(max_len, *item['timestamps'].shape[1:])
        
        # 复制原始数据
        forces_l_padded[:seq_len] = item['forces_l']
        forces_r_padded[:seq_len] = item['forces_r'] 
        action_padded[:seq_len] = item['action']
        timestamps_padded[:seq_len] = item['timestamps']
        
        batch_data['forces_l'].append(forces_l_padded)
        batch_data['forces_r'].append(forces_r_padded)
        batch_data['action'].append(action_padded)
        batch_data['timestamps'].append(timestamps_padded)
        batch_data['category'].append(item['category'])
    
    # 转换为tensor
    batch_data['forces_l'] = torch.stack(batch_data['forces_l'])
    batch_data['forces_r'] = torch.stack(batch_data['forces_r']) 
    batch_data['action'] = torch.stack(batch_data['action'])
    batch_data['timestamps'] = torch.stack(batch_data['timestamps'])
    batch_data['seq_lengths'] = torch.tensor(batch_data['seq_lengths'])
    
    return batch_data


def train_feature_mlp(config):
    """
    训练Feature-MLP模型
    Args:
        config: 配置字典
    """
    # 创建输出目录
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train_feature_mlp.log")
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print("Feature-MLP Training")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['lr']}")
    print(f"Feature Dim: {config['model']['feature_dim']}")
    print(f"Hidden Dims: {config['model']['hidden_dims']}")
    print("=" * 60)
    
    # 创建数据加载器
    print("创建数据加载器...")
    train_dataset, test_dataset = create_train_test_datasets(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        train_ratio=config['data']['train_split'],
        random_seed=42,  # 固定随机种子
        start_frame=config['data']['start_frame'],
        use_end_states=config['data']['load_end_effector'],
        use_forces=config['data']['load_forces'],
        use_resultants=config['data']['load_wrench'],
        normalize_config=config['data'].get('normalize_config', None)  # 支持归一化配置
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=False,  # 暂时禁用pin_memory
        persistent_workers=False,
        collate_fn=feature_mlp_collate_fn  # 使用Feature-MLP专用collate函数
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=0,  # 设置为0避免多进程问题
        pin_memory=False,  # 暂时禁用pin_memory
        persistent_workers=False,
        collate_fn=feature_mlp_collate_fn  # 使用Feature-MLP专用collate函数
    )
    
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = FeatureMLP(
        feature_dim=config['model']['feature_dim'],
        action_dim=config['model']['action_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model']['dropout_rate'],
        pretrained_encoder_path=config['model']['pretrained_encoder_path']
    ).cuda()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 创建优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    patience = config['training']['early_stopping_patience']
    
    # 记录损失历史
    loss_fields = ['main_loss', 'mae', 'mse', 'rmse']
    loss_history = {'epoch': [], 'total_loss': []}
    for field in loss_fields:
        loss_history[field] = []
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # 训练
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, config['loss'])
        
        # 评估
        if epoch % config['training']['eval_every'] == 0:
            eval_loss, eval_metrics = evaluate(model, test_loader, config['loss'])
        else:
            eval_loss, eval_metrics = train_loss, train_metrics
        
        # 学习率调度
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(eval_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练信息
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Eval Loss: {eval_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.6e}")
        print("  Train Metrics:")
        for key, value in train_metrics.items():
            if isinstance(value, list):
                print(f"    {key}: {value}")  # 直接打印列表
            else:
                print(f"    {key}: {value:.6f}")
        if epoch % config['training']['eval_every'] == 0:
            print("  Eval Metrics:")
            for key, value in eval_metrics.items():
                if isinstance(value, list):
                    print(f"    {key}: {value}")  # 直接打印列表
                else:
                    print(f"    {key}: {value:.6f}")
        print("-" * 50)
        
        # 记录损失历史
        loss_history['epoch'].append(epoch)
        loss_history['total_loss'].append(eval_loss)
        for key in loss_fields:
            if key in eval_metrics:
                loss_history[key].append(eval_metrics[key])
        
        # 早停检查
        if eval_loss < best_loss:
            best_loss = eval_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': eval_loss,
                'config': config
            }, os.path.join(output_dir, "best_model.pt"))
            print(f"保存最佳模型: best_loss = {best_loss:.6f}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"早停：{patience} 个epoch没有改善")
            break
        
        # 定期保存检查点
        if epoch % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': eval_loss,
                'config': config
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': eval_loss,
        'config': config
    }, os.path.join(output_dir, "final_model.pt"))
    
    # 保存训练损失曲线
    plot_all_losses_single_plot(
        loss_history, 
        save_path=os.path.join(output_dir, "training_loss_curves.png"),
        title="Feature-MLP Training Loss"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(output_dir, "loss_history.npy"), loss_history)
    
    print("✅ Feature-MLP训练完成!")
    print(f"最佳损失: {best_loss:.6f}")
    return model, loss_history


def train_epoch(model, train_loader, optimizer, loss_config):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    total_metrics = {}
    num_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        # collate函数已经将序列展开为单个时间步样本
        forces_l = batch['forces_l'].cuda()  # (N, 3, 20, 20)
        forces_r = batch['forces_r'].cuda()  # (N, 3, 20, 20)
        delta_targets = batch['delta'].cuda()  # (N, 3)
        
        if forces_l.size(0) == 0:  # 跳过空batch
            continue
        
        # 前向传播
        optimizer.zero_grad()
        predicted_deltas = model(forces_l, forces_r)
        
        # 计算损失
        loss, metrics = compute_feature_mlp_losses(
            predicted_deltas, delta_targets, loss_config
        )
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累计指标
        current_batch_size = forces_l.size(0)
        total_loss += loss.item() * current_batch_size
        num_samples += current_batch_size
        
        for key, value in metrics.items():
            # 跳过列表类型的指标，这些不需要累加
            if isinstance(value, list):
                continue
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value * current_batch_size
    
    # 计算平均指标
    if num_samples > 0:
        avg_loss = total_loss / num_samples
        avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
        # 添加最后一个batch的观察值
        if len(metrics) > 0:
            avg_metrics['last_prediction'] = metrics.get('last_prediction', [])
            avg_metrics['last_target'] = metrics.get('last_target', [])
    else:
        avg_loss = 0.0
        avg_metrics = {}
    
    return avg_loss, avg_metrics


def evaluate(model, test_loader, loss_config):
    """评估模型"""
    model.eval()
    
    total_loss = 0.0
    total_metrics = {}
    num_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # collate函数已经将序列展开为单个时间步样本
            forces_l = batch['forces_l'].cuda()  # (N, 3, 20, 20)
            forces_r = batch['forces_r'].cuda()  # (N, 3, 20, 20)
            delta_targets = batch['delta'].cuda()  # (N, 3)
            
            if forces_l.size(0) == 0:  # 跳过空batch
                continue
            
            # 前向传播
            predicted_deltas = model(forces_l, forces_r)
            
            # 计算损失
            loss, metrics = compute_feature_mlp_losses(
                predicted_deltas, delta_targets, loss_config
            )
            
            # 累计指标
            current_batch_size = forces_l.size(0)
            total_loss += loss.item() * current_batch_size
            num_samples += current_batch_size
            
            for key, value in metrics.items():
                # 跳过列表类型的指标，这些不需要累加
                if isinstance(value, list):
                    continue
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value * current_batch_size
    
    # 计算平均指标
    if num_samples > 0:
        avg_loss = total_loss / num_samples
        avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
        # 添加最后一个batch的观察值
        if len(metrics) > 0:
            avg_metrics['last_prediction'] = metrics.get('last_prediction', [])
            avg_metrics['last_target'] = metrics.get('last_target', [])
    else:
        avg_loss = 0.0
        avg_metrics = {}
    
    return avg_loss, avg_metrics


def main(config):
    """
    主训练函数
    """
    return train_feature_mlp(config)


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
            'train_split': 0.8,
            'batch_size': 16,
            'num_workers': 4,
            'start_frame': 0,
            'load_forces': True,
            'load_wrench': False,
            'load_end_effector': False
        },
        'model': {
            'feature_dim': 128,  # 匹配预训练权重的特征维度
            'action_dim': 3,
            'hidden_dims': [512, 512, 512],
            'dropout_rate': 0.1,
            'pretrained_encoder_path': 'tactile_representation/prototype_library/cnnae_crt_128.pt'  # 使用相对路径
        },
        'loss': {
            'mse_weight': 1.0,
            'l1_weight': 0.1,
            'l2_weight': 0.001
        },
        'training': {
            'epochs': 50,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'scheduler_step': 20,
            'scheduler_gamma': 0.5,
            'eval_every': 5,
            'save_every': 10,
            'early_stopping_patience': 15
        },
        'output': {
            'output_dir': os.path.join("./policy_learn/checkpoints", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_feature_mlp")
        }
    }
    
    main(config)
