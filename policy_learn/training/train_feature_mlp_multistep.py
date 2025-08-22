"""
Feature-MLP多步预测训练脚本

基于预训练触觉编码器的多步动作序列预测训练
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from policy_learn.models.feature_mlp_multistep import FeatureMLPMultiStep, compute_multistep_losses
from policy_learn.dataset_dataloader.flexible_policy_dataset import FlexiblePolicyDataset
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
    计算额外的指标用于与单步训练对比
    
    Args:
        predictions: (B, H, 3) 预测的动作序列
        targets: (B, H, 3) 真实的动作序列
        
    Returns:
        dict: 包含l1_error, rmse等指标
    """
    # 展平为 (B*H, 3)
    pred_flat = predictions.view(-1, predictions.size(-1))
    target_flat = targets.view(-1, targets.size(-1))
    
    # L1误差
    l1_error = torch.mean(torch.abs(pred_flat - target_flat)).item()
    
    # RMSE
    mse = torch.mean((pred_flat - target_flat) ** 2)
    rmse = torch.sqrt(mse).item()
    
    # 最后一组预测和目标（用于显示）
    last_prediction = predictions[-1, -1].detach().cpu().numpy()  # 最后一个样本的最后一步
    last_target = targets[-1, -1].detach().cpu().numpy()
    
    return {
        'l1_error': l1_error,
        'rmse': rmse,
        'last_prediction': last_prediction.tolist(),
        'last_target': last_target.tolist()
    }


def train_multistep_feature_mlp(config):
    """
    训练多步Feature-MLP模型
    
    Args:
        config: 训练配置字典
        
    Returns:
        str: 最佳模型保存路径
    """
    print("🎯 多步Feature-MLP训练开始")
    print("📊 配置摘要:")
    print(f"   数据根目录: {config['data']['data_root']}")
    print(f"   预测步数: {config['model']['horizon']}")
    print(f"   批次大小: {config['data']['batch_size']}")
    print(f"   学习率: {config['training']['lr']}")
    print(f"   训练轮数: {config['training']['epochs']}")
    print(f"   输出目录: {config['output']['output_dir']}")
    
    # 创建输出目录（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, config['output']['output_dir'], f"multistep_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 输出目录: {output_dir}")
    
    # 保存配置
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 初始化wandb
    if WANDB_AVAILABLE and config.get('wandb', {}).get('enabled', True):
        try:
            wandb.login()
            print("✅ wandb登录成功")
            
            wandb_config = config.get('wandb', {})
            run = wandb.init(
                mode=wandb_config.get('mode', 'online'),
                project=wandb_config.get('project', 'tactile-feature-mlp'),  # 与单步训练相同的项目
                entity=wandb_config.get('entity', None),
                name=f"feature_mlp_multistep_{timestamp}",  # 添加multistep后缀
                tags=wandb_config.get('tags', ['feature-mlp', 'multistep', 'tactile']),
                notes=wandb_config.get('notes', 'Multi-step Feature-MLP training with pretrained tactile encoder'),
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
    
    print("🚀 开始多步Feature-MLP训练...")
    
    # 1. 准备数据集
    print("\n📂 加载数据集...")
    
    horizon = config['model']['horizon']
    print(f"🔄 使用时序模式加载数据，序列长度: {horizon}")
    
    # 使用时序模式直接加载序列数据
    train_dataset = FlexiblePolicyDataset(
        data_root=os.path.join(project_root, config['data']['data_root']),
        categories=config['data']['categories'],
        is_train=True,
        train_ratio=config['data']['train_split'],
        start_frame=config['data']['start_frame'],
        normalize_config=config['data']['normalize_config'],
        sequence_mode=True,                    # 启用时序模式
        sequence_length=horizon                # 序列长度等于预测步数
    )
    
    val_dataset = FlexiblePolicyDataset(
        data_root=os.path.join(project_root, config['data']['data_root']),
        categories=config['data']['categories'],
        is_train=False,
        train_ratio=config['data']['train_split'],
        start_frame=config['data']['start_frame'],
        normalize_config=config['data']['normalize_config'],
        sequence_mode=True,                    # 启用时序模式
        sequence_length=horizon                # 序列长度等于预测步数
    )
    
    print(f"✅ 时序数据集准备完成")
    print(f"   训练样本: {len(train_dataset)}")
    print(f"   验证样本: {len(val_dataset)}")
    print(f"   每个样本包含 {horizon} 步连续数据")
    
    # 创建数据加载器
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
    
    # 2. 创建模型
    print("\n🧠 创建多步预测模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    model = FeatureMLPMultiStep(
        feature_dim=config['model']['feature_dim'],
        horizon=config['model']['horizon'],
        action_dim=config['model']['action_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model']['dropout_rate'],
        pretrained_encoder_path=os.path.join(project_root, config['model']['pretrained_encoder_path'])
    ).to(device)
    
    # 3. 设置优化器和调度器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # 训练循环
    best_loss = float('inf')
    best_model_path = None
    patience_counter = 0
    
    for epoch in range(config['training']['epochs']):
        print(f"\n🔄 Epoch {epoch+1}/{config['training']['epochs']}")
        
        # 训练阶段
        model.train()
        train_losses = []
        train_metrics = {
            'total_loss': [],
            'mse_loss': [],
            'l1_error': [],         # 添加L1误差，与单步训练对比
            'rmse': [],             # 添加RMSE，与单步训练对比
            'final_step_loss': [],
            'cumulative_error': []
        }
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
            # 时序模式数据格式: batch['forces_l/r']: (B, H, 3, 20, 20)
            #                 batch['actions']: (B, H, action_dim)
            
            # 取第一帧的触觉数据作为输入
            forces_l = batch['forces_l'][:, 0].to(device)  # (B, 3, 20, 20)
            forces_r = batch['forces_r'][:, 0].to(device)  # (B, 3, 20, 20)
            
            # 整个序列的动作作为目标，只取前3维位置增量 (dx, dy, dz)
            target_seq = batch['actions'][:, :, :3].to(device)  # (B, H, 3)
            
            optimizer.zero_grad()
            
            # 前向传播：用第一帧预测整个序列
            pred_seq = model(forces_l, forces_r)
            
            # 计算损失
            loss_dict = compute_multistep_losses(
                pred_seq, target_seq, config['loss']
            )
            
            # 计算额外指标（与单步训练对比）
            additional_metrics = compute_additional_metrics(pred_seq, target_seq)
            
            loss = loss_dict['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（与单步训练保持一致）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_losses.append(loss.item())
            for key in train_metrics:
                if key in loss_dict:
                    # 确保张量转换为CPU标量
                    if isinstance(loss_dict[key], torch.Tensor):
                        train_metrics[key].append(loss_dict[key].detach().cpu().item())
                    else:
                        train_metrics[key].append(loss_dict[key])
                elif key in additional_metrics:
                    # 添加额外指标
                    train_metrics[key].append(additional_metrics[key])
        
        # 计算平均训练指标
        avg_train_loss = np.mean(train_losses)
        avg_train_metrics = {k: np.mean(v) for k, v in train_metrics.items()}
        
        # 打印训练结果（与单步训练格式一致）
        print(f"🔄 训练: Loss={avg_train_loss:.6f}, L1={avg_train_metrics.get('l1_error', 0):.6f}, RMSE={avg_train_metrics.get('rmse', 0):.6f}")
        
        # 验证阶段
        if epoch % config['training']['eval_every'] == 0:
            model.eval()
            val_losses = []
            val_metrics = {
                'total_loss': [],
                'mse_loss': [],
                'l1_error': [],         # 添加L1误差，与单步训练对比
                'rmse': [],             # 添加RMSE，与单步训练对比
                'final_step_loss': [],
                'cumulative_error': [],
                'last_prediction': None,  # 最后一组预测
                'last_target': None       # 最后一组目标
            }
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Evaluating"):
                    # 取第一帧的触觉数据作为输入
                    forces_l = batch['forces_l'][:, 0].to(device)  # (B, 3, 20, 20)
                    forces_r = batch['forces_r'][:, 0].to(device)  # (B, 3, 20, 20)
                    
                    # 整个序列的动作作为目标，只取前3维位置增量 (dx, dy, dz)
                    target_seq = batch['actions'][:, :, :3].to(device)  # (B, H, 3)
                    
                    pred_seq = model(forces_l, forces_r)
                    loss_dict = compute_multistep_losses(
                        pred_seq, target_seq, config['loss']
                    )
                    
                    # 计算额外指标（与单步训练对比）
                    additional_metrics = compute_additional_metrics(pred_seq, target_seq)
                    
                    val_losses.append(loss_dict['total_loss'].detach().cpu().item())
                    for key in val_metrics:
                        if key in loss_dict:
                            # 确保张量转换为CPU标量
                            if isinstance(loss_dict[key], torch.Tensor):
                                val_metrics[key].append(loss_dict[key].detach().cpu().item())
                            else:
                                val_metrics[key].append(loss_dict[key])
                        elif key in additional_metrics:
                            # 添加额外指标或保留最后一组值
                            if key in ['last_prediction', 'last_target']:
                                val_metrics[key] = additional_metrics[key]
                            else:
                                val_metrics[key].append(additional_metrics[key])
            
            avg_val_loss = np.mean(val_losses)
            avg_val_metrics = {k: np.mean(v) if isinstance(v, list) else v for k, v in val_metrics.items()}
            
            print(f"📊 验证结果:")
            print(f"   Loss: {avg_val_loss:.6f}")
            print(f"   L1: {avg_val_metrics.get('l1_error', 0):.6f}")
            print(f"   RMSE: {avg_val_metrics.get('rmse', 0):.6f}")
            print(f"   Final step loss: {avg_val_metrics.get('final_step_loss', 0):.6f}")
            print(f"   Last prediction: {avg_val_metrics.get('last_prediction', [])}")
            print(f"   Last target: {avg_val_metrics.get('last_target', [])}")
            
            # 学习率调度
            scheduler.step(avg_val_loss)
            
            # 记录到wandb (与单步训练相同格式)
            if use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': avg_train_loss,
                    'train/l1_error': avg_train_metrics.get('l1_error', 0),
                    'train/rmse': avg_train_metrics.get('rmse', 0),
                    'val/loss': avg_val_loss,
                    'val/l1_error': avg_val_metrics.get('l1_error', 0),
                    'val/rmse': avg_val_metrics.get('rmse', 0),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    # 多步特有的指标
                    'train/final_step': avg_train_metrics.get('final_step_loss', 0),
                    'val/final_step': avg_val_metrics.get('final_step_loss', 0),
                    'val/cumulative_error': avg_val_metrics.get('cumulative_error', 0)
                }
                wandb.log(log_dict)
            
            # 保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                
                best_model_path = os.path.join(output_dir, 'best_multistep_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'config': config
                }, best_model_path)
                print(f"💾 保存最佳模型: {best_model_path}")
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= config['training']['patience']:
                print(f"🛑 早停触发，已等待{patience_counter}轮无改善")
                break
        
        # 定期保存检查点
        if epoch % config['training']['save_every'] == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'config': config
            }, checkpoint_path)
    
    # 训练完成
    if use_wandb:
        wandb.finish()
    
    print(f"\n🎉 多步训练完成！")
    print(f"📈 最佳验证损失: {best_loss:.4f}")
    print(f"💾 最佳模型: {best_model_path}")
    
    return best_model_path


def main(config=None):
    """主函数"""
    if config is None:
        # 默认配置
        config = {
            'data': {
                'data_root': 'datasets/data25.7_aligned',
                'categories': [
                    "cir_lar", "cir_med", "cir_sma",
                    "rect_lar", "rect_med", "rect_sma", 
                    "tri_lar", "tri_med", "tri_sma"
                ],
                'train_split': 0.8,
                'batch_size': 16,  # 多步预测数据更复杂，减小batch size
                'num_workers': 4,
                'start_frame': 0,
                'normalize_config': {
                    'forces': 'zscore',
                    'actions': 'minmax',
                    'end_states': None,
                    'resultants': None
                }
            },
            'model': {
                'feature_dim': 128,
                'horizon': 5,      # 预测5步
                'action_dim': 3,   # (dx, dy, dz)
                'hidden_dims': [256, 128],
                'dropout_rate': 0.25,
                'pretrained_encoder_path': 'tactile_representation/prototype_library/cnnae_crt_128.pt'
            },
            'loss': {
                'loss_type': 'huber',
                'huber_delta': 1.0,
                'step_weights': [2.0, 1.5, 1.2, 1.0, 1.0]  # 等权重，或者可以设置如 [1.0, 1.2, 1.5, 2.0, 3.0] 重点关注后续步骤
            },
            'training': {
                'epochs': 100,
                'lr': 5e-5,    # 多步预测使用较小学习率
                'weight_decay': 1e-4,
                'eval_every': 2,
                'save_every': 10,
                'patience': 20
            },
            'output': {
                'output_dir': 'policy_learn/checkpoints'
            },
            'wandb': {
                'enabled': True,
                'mode': 'online',
                'project': 'tactile-feature-mlp',  # 与单步训练相同项目
                'entity': None,
                'tags': ['feature-mlp', 'multistep', 'tactile', 'behavior-cloning'],  # 与单步训练相似标签
                'notes': 'Multi-step Feature-MLP training with pretrained tactile encoder and flexible dataset'
            }
        }
    
    # 检查路径
    data_path = os.path.join(project_root, config['data']['data_root'])
    pretrained_path = os.path.join(project_root, config['model']['pretrained_encoder_path'])
    
    print(f"🎯 多步Feature-MLP训练配置:")
    print(f"   项目根目录: {project_root}")
    print(f"   数据路径: {data_path}")
    print(f"   预训练权重: {pretrained_path}")
    print(f"   预测步数: {config['model']['horizon']}")
    print(f"   输出目录: {os.path.join(project_root, config['output']['output_dir'])}")
    
    if os.path.exists(data_path):
        print(f"✅ 数据路径存在")
    else:
        print(f"❌ 数据路径不存在: {data_path}")
        return
    
    if os.path.exists(pretrained_path):
        print(f"✅ 预训练权重存在")
    else:
        print(f"⚠️  预训练权重不存在: {pretrained_path}")
        print("将使用随机初始化")
        config['model']['pretrained_encoder_path'] = ''
    
    # 开始训练
    print("\n" + "="*60)
    best_model_path = train_multistep_feature_mlp(config)
    print("="*60)
    
    return best_model_path


if __name__ == '__main__':
    main()
