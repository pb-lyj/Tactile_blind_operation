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
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
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
        use_resultants=config['data']['load_wrench']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
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
    patience = config['training']['patience']
    
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
            print(f"    {key}: {value:.6f}")
        if epoch % config['training']['eval_every'] == 0:
            print("  Eval Metrics:")
            for key, value in eval_metrics.items():
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
        # 获取数据
        forces_l = batch['forces_l'].cuda()  # (B, T, 3, 20, 20)
        forces_r = batch['forces_r'].cuda()  # (B, T, 3, 20, 20)
        actions = batch['action'].cuda()     # (B, T, 6)
        
        # 只使用位置信息 (前3维)
        positions = actions[:, :, :3]  # (B, T, 3)
        
        # 创建输入-目标对
        batch_size, seq_len = positions.shape[:2]
        
        # 收集所有时间步的数据
        all_forces_l = []
        all_forces_r = []
        all_deltas = []
        
        for t in range(seq_len - 1):  # 最后一帧没有下一帧
            # 当前帧的触觉数据
            curr_forces_l = forces_l[:, t]  # (B, 3, 20, 20)
            curr_forces_r = forces_r[:, t]  # (B, 3, 20, 20)
            
            # 计算动作增量
            curr_pos = positions[:, t]      # (B, 3)
            next_pos = positions[:, t + 1]  # (B, 3)
            delta = next_pos - curr_pos     # (B, 3)
            
            all_forces_l.append(curr_forces_l)
            all_forces_r.append(curr_forces_r)
            all_deltas.append(delta)
        
        if len(all_forces_l) == 0:
            continue
        
        # 合并所有时间步
        forces_l_input = torch.cat(all_forces_l, dim=0)  # (B*(T-1), 3, 20, 20)
        forces_r_input = torch.cat(all_forces_r, dim=0)  # (B*(T-1), 3, 20, 20)
        delta_targets = torch.cat(all_deltas, dim=0)     # (B*(T-1), 3)
        
        # 前向传播
        optimizer.zero_grad()
        predicted_deltas = model(forces_l_input, forces_r_input)
        
        # 计算损失
        loss, metrics = compute_feature_mlp_losses(
            predicted_deltas, delta_targets, loss_config
        )
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累计指标
        current_batch_size = forces_l_input.size(0)
        total_loss += loss.item() * current_batch_size
        num_samples += current_batch_size
        
        for key, value in metrics.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value * current_batch_size
    
    # 计算平均指标
    if num_samples > 0:
        avg_loss = total_loss / num_samples
        avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
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
            # 获取数据
            forces_l = batch['forces_l'].cuda()
            forces_r = batch['forces_r'].cuda()
            actions = batch['action'].cuda()
            
            positions = actions[:, :, :3]
            batch_size, seq_len = positions.shape[:2]
            
            # 收集所有时间步的数据
            all_forces_l = []
            all_forces_r = []
            all_deltas = []
            
            for t in range(seq_len - 1):
                curr_forces_l = forces_l[:, t]
                curr_forces_r = forces_r[:, t]
                
                curr_pos = positions[:, t]
                next_pos = positions[:, t + 1]
                delta = next_pos - curr_pos
                
                all_forces_l.append(curr_forces_l)
                all_forces_r.append(curr_forces_r)
                all_deltas.append(delta)
            
            if len(all_forces_l) == 0:
                continue
            
            # 合并数据
            forces_l_input = torch.cat(all_forces_l, dim=0)
            forces_r_input = torch.cat(all_forces_r, dim=0)
            delta_targets = torch.cat(all_deltas, dim=0)
            
            # 前向传播
            predicted_deltas = model(forces_l_input, forces_r_input)
            
            # 计算损失
            loss, metrics = compute_feature_mlp_losses(
                predicted_deltas, delta_targets, loss_config
            )
            
            # 累计指标
            current_batch_size = forces_l_input.size(0)
            total_loss += loss.item() * current_batch_size
            num_samples += current_batch_size
            
            for key, value in metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value * current_batch_size
    
    # 计算平均指标
    if num_samples > 0:
        avg_loss = total_loss / num_samples
        avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}
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
    main()
