"""
MLP策略模型训练脚本
基于flexible_policy_dataset.py数据处理和低维策略要求
输入: resultant_force[6] + resultant_moment[6]
输出: delta_action_nextstep[3]
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.logging import Logger
from utils.visualization import plot_all_losses_single_plot
from policy_learn.dataset_dataloader.flexible_policy_dataset import create_flexible_datasets
from policy_learn.models.mlp import create_tactile_policy_mlp, compute_mlp_policy_losses


def prepare_mlp_input_from_flexible_dataset(batch_data):
    """
    从FlexiblePolicyDataset批次中准备MLP模型的输入
    
    Args:
        batch_data: 来自FlexiblePolicyDataset的批次数据
    
    Returns:
        dict: MLP模型的输入字典
    """
    # 合并左右手的合力和合力矩
    resultant_force = torch.cat([
        batch_data['resultant_force_l'], 
        batch_data['resultant_force_r']
    ], dim=-1)  # (B, 6) - 无时序模式
    
    resultant_moment = torch.cat([
        batch_data['resultant_moment_l'], 
        batch_data['resultant_moment_r']
    ], dim=-1)  # (B, 6) - 无时序模式
    
    # 使用数据集计算的真实动作增量
    target_delta_action = batch_data['action'][:,:3]  # (B, 3) - 来自数据集的真实增量
    # print(target_delta_action.shape)
    return {
        'resultant_force': resultant_force,
        'resultant_moment': resultant_moment,
        'target_delta_action': target_delta_action
    }


def train_mlp_policy(config):
    """
    训练MLP策略模型
    Args:
        config: 配置字典
    """
    # 创建输出目录
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train_mlp_policy.log")
    sys.stdout = Logger(log_file)
    
    print("=" * 60)
    print("MLP Policy Training (Flexible Dataset)")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['lr']}")
    print("=" * 60)
    
    # 创建训练和测试数据集（无时序模式）
    train_dataset, test_dataset = create_flexible_datasets(
        data_root=config['data']['data_root'],
        categories=config['data']['categories'],
        train_ratio=config['data']['train_ratio'],
        random_seed=config['data']['random_seed'],
        start_frame=config['data']['start_frame'],
        use_end_states=config['data']['use_end_states'],
        use_forces=config['data']['use_forces'],
        use_resultants=config['data']['use_resultants'],
        normalize_config=config['data'].get('normalize_config', None),
        sequence_mode=False,  # MLP使用无时序模式
        sequence_length=1     # 单帧数据
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=config['data']['num_workers'], 
        pin_memory=True
    )

    # 创建模型
    model = create_tactile_policy_mlp(config['model']).cuda()
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['training']['lr'], 
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # 训练循环
    best_loss = float('inf')
    patience_counter = 0
    patience = config['training']['patience']
    
    # 记录损失历史
    train_history = {'epoch': [], 'loss': []}
    test_history = {'epoch': [], 'loss': []}

    for epoch in range(1, config['training']['epochs'] + 1):
        # 训练阶段
        model.train()
        train_total_loss = 0
        train_samples = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} [Train]"):
            # 准备输入数据
            mlp_inputs = prepare_mlp_input_from_flexible_dataset(batch)
            
            # 将数据移到GPU
            for key in mlp_inputs:
                if isinstance(mlp_inputs[key], torch.Tensor):
                    mlp_inputs[key] = mlp_inputs[key].cuda()
            
            # 前向传播
            action = mlp_inputs['target_delta_action']
            # print(action.shape)
            resultant_force = mlp_inputs['resultant_force']  # (B, 6)
            resultant_moment = mlp_inputs['resultant_moment']  # (B, 6)
            # print(resultant_force.shape)
            outputs = model(resultant_force, resultant_moment)
            
            # 计算损失
            loss, metrics = compute_mlp_policy_losses(mlp_inputs, outputs, config['loss'])
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累积损失
            batch_size = resultant_force.size(0)
            train_total_loss += loss.item() * batch_size
            train_samples += batch_size
        
        # 计算训练平均损失
        train_avg_loss = train_total_loss / train_samples
        
        # 测试阶段
        model.eval()
        test_total_loss = 0
        test_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} [Test]"):
                # 准备输入数据
                mlp_inputs = prepare_mlp_input_from_flexible_dataset(batch)
                
                # 将数据移到GPU
                for key in mlp_inputs:
                    if isinstance(mlp_inputs[key], torch.Tensor):
                        mlp_inputs[key] = mlp_inputs[key].cuda()
                
                # 前向传播
                resultant_force = mlp_inputs['resultant_force']  # (B, 6)
                resultant_moment = mlp_inputs['resultant_moment']  # (B, 6)
                
                outputs = model(resultant_force, resultant_moment)
                
                # 计算损失
                loss, metrics = compute_mlp_policy_losses(mlp_inputs, outputs, config['loss'])
                
                # 累积损失
                batch_size = resultant_force.size(0)
                test_total_loss += loss.item() * batch_size
                test_samples += batch_size
        
        # 计算测试平均损失
        test_avg_loss = test_total_loss / test_samples
        
        # 学习率调度
        scheduler.step(train_total_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练信息
        print(test_total_loss)
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"  Train Loss: {train_avg_loss:.6f}")
        print(f"  Test Loss: {test_avg_loss:.6f}")
        print(f"  Learning Rate: {current_lr:.6e}")
        print("-" * 50)
        
        # 记录损失历史
        train_history['epoch'].append(epoch)
        train_history['loss'].append(train_avg_loss)
        test_history['epoch'].append(epoch)
        test_history['loss'].append(test_avg_loss)
        
        # 早停检查
        if test_avg_loss < best_loss:
            best_loss = test_avg_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': train_avg_loss,
                'test_loss': test_avg_loss,
                'config': config
            }, os.path.join(output_dir, "best_model.pt"))
            print(f"✅ 保存最佳模型 (Test Loss: {best_loss:.6f})")
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
        'train_loss': train_avg_loss,
        'test_loss': test_avg_loss,
        'config': config
    }, os.path.join(output_dir, "final_model.pt"))
    
    # 保存训练损失曲线
    combined_history = {
        'epoch': train_history['epoch'],
        'train_loss': train_history['loss'],
        'test_loss': test_history['loss']
    }
    
    plot_all_losses_single_plot(
        combined_history, 
        save_path=os.path.join(output_dir, "training_loss_curves.png"),
        title="MLP Policy Training Loss (Flexible Dataset)"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(output_dir, "train_loss_history.npy"), train_history)
    np.save(os.path.join(output_dir, "test_loss_history.npy"), test_history)
    
    print("✅ MLP策略模型训练完成!")
    return model, train_history, test_history


if __name__ == '__main__':
    # 默认配置
    config = {
        'data': {
            'data_root': '/home/leonhard/tactile_operation_blind_stack/Tactile_blind_operation/datasets/data25.7_aligned',
            'categories': [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ],
            'train_ratio': 0.8,
            'random_seed': 42,
            'start_frame': 0,
            'use_end_states': True,
            'use_forces': False,  # MLP不需要原始力数据
            'use_resultants': True,  # 需要合力/矩数据
            'normalize_config': {
                'actions': 'minmax',
                'resultants': 'zscore',  # z-score标准化合力/矩数据
                'end_states': None
            },
            'num_workers': 8
        },
        'model': {
            'input_dim': 12,  # resultant_force[6] + resultant_moment[6]
            'output_dim': 3,  # delta_action_nextstep[3]
            'hidden_dim': 256,
            'num_layers': 3,
            'dropout': 0.1,
            'use_normalization': True  # z-score标准化
        },
        'loss': {
            'l2_weight': 1.0,        # L2损失权重（主要损失）
            'huber_weight': 0.0      # Huber损失权重（可选，设为0表示不使用）
        },
        'training': {
            'batch_size': 32,
            # 'epochs': 100,
            'epochs':60,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'patience': 20
        },
        'output': {
            'output_dir': os.path.join("./policy_models", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_mlp_policy_flexible")
        }
    }
    
    train_mlp_policy(config)