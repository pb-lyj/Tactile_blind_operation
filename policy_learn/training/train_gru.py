"""
GRU时序策略模型训练脚本
基于flexible_policy_dataset.py数据处理和低维策略（时序）要求
输入: resultant_force[6] + resultant_moment[6] {t}
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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.logging import Logger
from utils.visualization import plot_all_losses_single_plot
from policy_learn.dataset_dataloader.flexible_policy_dataset import create_flexible_datasets
from policy_learn.models.gru import create_tactile_policy_gru, compute_gru_policy_losses

# 设置代理（如果需要代理才能访问外网）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 设置超时时间
os.environ["WANDB_HTTP_TIMEOUT"] = "60"
# wandb导入和错误处理
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("⚠️  wandb未安装，将跳过在线日志记录")
    WANDB_AVAILABLE = False


def compute_additional_metrics(predictions, targets):
    """
    计算额外的指标用于wandb记录
    
    Args:
        predictions: 预测的动作序列 (B, T-1, 3)
        targets: 真实的动作序列 (B, T-1, 3)
        
    Returns:
        dict: 包含l1_error等指标
    """
    # 展平为 (B*(T-1), 3)
    pred_flat = predictions.view(-1, predictions.size(-1))
    target_flat = targets.view(-1, targets.size(-1))
    
    # L1误差
    l1_error = torch.mean(torch.abs(pred_flat - target_flat)).item()
    
    # 最后一组预测和目标（用于显示）
    last_prediction = predictions[-1, -1].detach().cpu().numpy()  # 最后一个样本的最后一步
    last_target = targets[-1, -1].detach().cpu().numpy()
    
    return {
        'l1_error': l1_error,
        'last_prediction': last_prediction.tolist(),
        'last_target': last_target.tolist()
    }
    
    
def prepare_gru_input_from_flexible_dataset(batch_data):
    """
    从FlexiblePolicyDataset批次中准备GRU模型的输入
    
    Args:
        batch_data: 来自FlexiblePolicyDataset的批次数据（时序模式）
    
    Returns:
        dict: GRU模型的输入字典
    """
    # 合并左右手的合力和合力矩
    resultant_force = torch.cat([
        batch_data['resultant_force_l'], 
        batch_data['resultant_force_r']
    ], dim=-1)  # (B, T, 6) - 时序模式
    
    resultant_moment = torch.cat([
        batch_data['resultant_moment_l'], 
        batch_data['resultant_moment_r']
    ], dim=-1)  # (B, T, 6) - 时序模式
    
    # 计算目标动作增量序列
    actions = batch_data['actions']  # (B, T, action_dim) - 注意flexible_policy_dataset时序模式返回'actions'而不是'action'
    current_actions = actions[:, :-1, :3]  # 前T-1步的位置 (B, T-1, 3)
    next_actions = actions[:, 1:, :3]  # 后T-1步的位置 (B, T-1, 3)
    target_delta_action = next_actions - current_actions  # (B, T-1, 3)
    
    # 对应调整输入特征（取前T-1步）
    resultant_force = resultant_force[:, :-1]  # (B, T-1, 6)
    resultant_moment = resultant_moment[:, :-1]  # (B, T-1, 6)
    
    return {
        'resultant_force': resultant_force,
        'resultant_moment': resultant_moment,
        'target_delta_action': target_delta_action
    }


def train_tactile_policy_gru(config):
    """
    训练GRU策略模型
    Args:
        config: 配置字典
    """
    # 创建输出目录
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train_tactile_policy_gru.log")
    sys.stdout = Logger(log_file)
    
    # 初始化wandb
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if WANDB_AVAILABLE and config.get('wandb', {}).get('enabled', True):
        try:
            wandb.login()
            print("✅ wandb登录成功")
            
            wandb_config = config.get('wandb', {})
            run = wandb.init(
                mode=wandb_config.get('mode', 'online'),
                project=wandb_config.get('project', 'tactile-gru-policy'),
                entity=wandb_config.get('entity', None),
                name=f"gru_policy_{timestamp}",
                tags=wandb_config.get('tags', ['gru', 'policy', 'tactile']),
                notes=wandb_config.get('notes', 'GRU policy training with flexible dataset'),
                config=config,
                dir=output_dir
            )
            print("📊 wandb初始化成功")
            use_wandb = True
        except Exception as e:
            print(f"⚠️  wandb初始化失败: {e}")
            use_wandb = False
    else:
        use_wandb = False
    
    print("=" * 60)
    print("GRU Policy Training (Flexible Dataset)")
    print(f"Output Directory: {output_dir}")
    print(f"Data Root: {config['data']['data_root']}")
    print(f"Sequence Length: {config['data']['sequence_length']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['training']['lr']}")
    print("=" * 60)
    
    # 创建训练和测试数据集（时序模式）
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
        sequence_mode=True,  # GRU使用时序模式
        sequence_length=config['data']['sequence_length']
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 检查数据集是否为空
    if len(train_dataset) == 0:
        print("❌ 错误：训练集为空！")
        print(f"   数据路径: {config['data']['data_root']}")
        print(f"   数据类别: {config['data']['categories']}")
        print("   请检查数据路径和类别设置")
        return None, None, None
        
    if len(test_dataset) == 0:
        print("❌ 错误：测试集为空！")
        print("   请检查train_ratio设置或数据量")
        return None, None, None
    
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
    model = create_tactile_policy_gru(config['model']).cuda()
    
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
    train_history = {'epoch': [], 'loss': []}
    test_history = {'epoch': [], 'loss': []}

    for epoch in range(1, config['training']['epochs'] + 1):
        # 训练阶段
        model.train()
        train_total_loss = 0
        train_samples = 0
        train_l1_errors = []
        train_last_predictions = []
        train_last_targets = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} [Train]"):
            # 准备输入数据
            gru_inputs = prepare_gru_input_from_flexible_dataset(batch)
            
            # 将数据移到GPU
            for key in gru_inputs:
                if isinstance(gru_inputs[key], torch.Tensor):
                    gru_inputs[key] = gru_inputs[key].cuda()
            
            # 前向传播
            resultant_force = gru_inputs['resultant_force']  # (B, T-1, 6)
            resultant_moment = gru_inputs['resultant_moment']  # (B, T-1, 6)
            
            # 初始化隐藏状态
            batch_size = resultant_force.size(0)
            hidden = model.init_hidden(batch_size, resultant_force.device)
            
            outputs, _ = model(resultant_force, resultant_moment, hidden)
            
            # 计算损失
            loss, metrics = compute_gru_policy_losses(gru_inputs, outputs, config['loss'])
            
            # 计算额外指标
            additional_metrics = compute_additional_metrics(outputs, gru_inputs['target_delta_action'])
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 累积损失和指标
            current_batch_size = resultant_force.size(0) * resultant_force.size(1)  # B * (T-1)
            train_total_loss += loss.item() * current_batch_size
            train_samples += current_batch_size
            train_l1_errors.append(additional_metrics['l1_error'])
            train_last_predictions.append(additional_metrics['last_prediction'])
            train_last_targets.append(additional_metrics['last_target'])
        
        # 计算训练平均损失和指标
        train_avg_loss = train_total_loss / train_samples
        train_avg_l1 = np.mean(train_l1_errors)
        train_last_pred = train_last_predictions[-1] if train_last_predictions else None
        train_last_tgt = train_last_targets[-1] if train_last_targets else None
        
        # 测试阶段
        model.eval()
        test_total_loss = 0
        test_samples = 0
        test_l1_errors = []
        test_last_predictions = []
        test_last_targets = []
        all_predictions = []  # 收集所有预测值
        all_targets = []      # 收集所有目标值
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} [Test]"):
                # 准备输入数据
                gru_inputs = prepare_gru_input_from_flexible_dataset(batch)
                
                # 将数据移到GPU
                for key in gru_inputs:
                    if isinstance(gru_inputs[key], torch.Tensor):
                        gru_inputs[key] = gru_inputs[key].cuda()
                
                # 前向传播
                resultant_force = gru_inputs['resultant_force']  # (B, T-1, 6)
                resultant_moment = gru_inputs['resultant_moment']  # (B, T-1, 6)
                
                # 初始化隐藏状态
                batch_size = resultant_force.size(0)
                hidden = model.init_hidden(batch_size, resultant_force.device)
                
                outputs, _ = model(resultant_force, resultant_moment, hidden)
                
                # 计算损失
                loss, metrics = compute_gru_policy_losses(gru_inputs, outputs, config['loss'])
                
                # 计算额外指标
                additional_metrics = compute_additional_metrics(outputs, gru_inputs['target_delta_action'])
                
                # 收集所有预测和目标值 (展平为 (B*(T-1), 3))
                pred_flat = outputs.view(-1, outputs.size(-1)).detach().cpu().numpy()  # (B*(T-1), 3)
                target_flat = gru_inputs['target_delta_action'].view(-1, gru_inputs['target_delta_action'].size(-1)).detach().cpu().numpy()  # (B*(T-1), 3)
                all_predictions.extend(pred_flat.tolist())
                all_targets.extend(target_flat.tolist())
                
                # 累积损失和指标
                current_batch_size = resultant_force.size(0) * resultant_force.size(1)  # B * (T-1)
                test_total_loss += loss.item() * current_batch_size
                test_samples += current_batch_size
                test_l1_errors.append(additional_metrics['l1_error'])
                test_last_predictions.append(additional_metrics['last_prediction'])
                test_last_targets.append(additional_metrics['last_target'])
        
        # 计算测试平均损失和指标
        test_avg_loss = test_total_loss / test_samples
        test_avg_l1 = np.mean(test_l1_errors)
        test_last_pred = test_last_predictions[-1] if test_last_predictions else None
        test_last_tgt = test_last_targets[-1] if test_last_targets else None
        
        # 学习率调度
        scheduler.step(test_avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印训练信息
        print(f"Epoch {epoch}/{config['training']['epochs']}")
        print(f"  Train Loss: {train_avg_loss:.6f}, L1: {train_avg_l1:.6f}")
        print(f"  Test Loss: {test_avg_loss:.6f}, L1: {test_avg_l1:.6f}")
        print(f"  Learning Rate: {current_lr:.6e}")
        print(f"  Last prediction: {test_last_pred}")
        print(f"  Last target: {test_last_tgt}")
        print("-" * 50)
        
        # 输出所有target和prediction值到log文件
        print("=== All Target and Prediction Values ===")
        for i, (target, prediction) in enumerate(zip(all_targets, all_predictions)):
            # 每行格式: target[0] target[1] target[2] prediction[0] prediction[1] prediction[2]
            print(f"{target[0]:.6f} {target[1]:.6f} {target[2]:.6f} {prediction[0]:.6f} {prediction[1]:.6f} {prediction[2]:.6f}")
        print("=== End of Target and Prediction Values ===")
        
        # 记录到wandb
        if use_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_avg_loss,
                'train/l1_error': train_avg_l1,
                'test/loss': test_avg_loss,
                'test/l1_error': test_avg_l1,
                'learning_rate': current_lr
            }
            # 添加最后一组预测和目标（如果存在）
            if test_last_pred is not None:
                log_dict['test/last_prediction'] = test_last_pred
            if test_last_tgt is not None:
                log_dict['test/last_target'] = test_last_tgt
            wandb.log(log_dict)
        
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
        title="GRU Policy Training Loss (Flexible Dataset)"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(output_dir, "train_loss_history.npy"), train_history)
    np.save(os.path.join(output_dir, "test_loss_history.npy"), test_history)
    
    # 结束wandb记录
    if use_wandb:
        wandb.finish()
    
    print("✅ GRU策略模型训练完成!")
    return model, train_history, test_history


if __name__ == '__main__':
    # 默认配置
    config = {
        'data': {
            'data_root': os.path.join(project_root, 'datasets', 'data25.7_aligned'),  # 使用项目根目录 + 相对路径
            'categories': [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ],
            'sequence_length': 10,
            'train_ratio': 0.8,
            'random_seed': 42,
            'start_frame': 0,
            'use_end_states': True,
            'use_forces': False,  # GRU不需要原始力数据
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
            'num_layers': 2,
            'dropout': 0.1,
            'use_normalization': True  # z-score标准化
        },
        'loss': {
            'l2_weight': 1.0,           # L2损失权重（主要损失）
            'huber_weight': 0.0,        # Huber损失权重（可选，设为0表示不使用）
            'delta_u_weight': 0.1,      # Δu平滑正则权重
            'jerk_weight': 0.05         # jerk平滑正则权重
        },
        'training': {
            'batch_size': 16,
            # 'epochs': 100,
            'epochs': 50,
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 20
        },
        'output': {
            'output_dir': os.path.join(project_root, "policy_learn/checkpoints", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_gru_policy")
        },
        'wandb': {
            'enabled': True,
            'mode': 'online',
            'project': 'tactile-feature-mlp',
            'entity': None,
            'tags': ['gru', 'policy', 'tactile', 'flexible-dataset'],
            'notes': 'GRU policy training with flexible dataset and L1 metrics'
        }
    }
    
    train_tactile_policy_gru(config)