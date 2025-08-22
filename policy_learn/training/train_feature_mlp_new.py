"""
Feature-MLP训练脚本 - 使用FlexiblePolicyDataset
基于预训练触觉特征的行为克隆训练
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime
import wandb

# 获取项目根路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

try:
    from tactile_representation.Prototype_Discovery.models.cnn_autoencoder import TactileCNNAutoencoder
except ImportError:
    print("警告: 无法导入 TactileCNNAutoencoder")
    TactileCNNAutoencoder = None
from policy_learn.models.feature_mlp_new import FeatureMLP, compute_feature_mlp_losses
from policy_learn.dataset_dataloader.flexible_policy_dataset import create_flexible_datasets

# 设置代理（如果需要代理才能访问外网）
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# 设置超时时间
os.environ["WANDB_HTTP_TIMEOUT"] = "60"

def train_feature_mlp(config):
    """
    训练Feature-MLP模型
    
    Args:
        config: 训练配置字典
        
    Returns:
        best_model_path: 最佳模型权重路径
    """
    print("🚀 开始Feature-MLP训练...")
    
    # 登录wandb（如果需要的话）
    try:
        wandb.login()
        print("✅ wandb登录成功")
    except Exception as e:
        print(f"⚠️  wandb登录警告: {e}")
        print("继续训练，但可能无法上传到wandb")
    
    # 创建带时间戳的输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(project_root, config['output']['output_dir'], f"feature_mlp_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化wandb
    run = wandb.init(
        entity=config.get('wandb', {}).get('entity', None),
        project=config.get('wandb', {}).get('project', 'feature-mlp-training'),
        name=f"feature_mlp_{timestamp}",
        config=config,
        dir=output_dir,
        tags=config.get('wandb', {}).get('tags', ['feature-mlp', 'tactile'])
    )
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 创建数据集
    print("📂 加载数据集...")
    normalize_config = config['data'].get('normalize_config', {
        'forces': 'zscore',
        'actions': 'minmax',
        'end_states': None,
        'resultants': None
    })
    
    train_dataset, test_dataset = create_flexible_datasets(
        data_root=os.path.join(project_root, config['data']['data_root']),
        categories=config['data'].get('categories', None),
        train_ratio=config['data']['train_split'],
        random_seed=42,
        start_frame=config['data'].get('start_frame', 0),
        use_end_states=False,  # Feature-MLP不需要状态信息
        use_forces=True,       # 需要触觉力数据
        use_resultants=False,  # 不需要合力数据
        normalize_config=normalize_config,
        sequence_mode=False    # 使用无时序模式
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"✅ 训练集: {len(train_dataset)} 样本")
    print(f"✅ 测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    print("🏗️ 创建模型...")
    pretrained_path = os.path.join(project_root, config['model']['pretrained_encoder_path']) if config['model']['pretrained_encoder_path'] else None
    model = FeatureMLP(
        feature_dim=config['model']['feature_dim'],
        action_dim=config['model']['action_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout_rate=config['model']['dropout_rate'],
        pretrained_encoder_path=pretrained_path
    )
    model = model.to(device)
    
    # 创建优化器和学习率调度器
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
    patience_counter = 0
    
    try:
        for epoch in range(config['training']['epochs']):
            print(f"\n🔄 Epoch {epoch + 1}/{config['training']['epochs']}")
            
            # 训练阶段
            train_loss, train_metrics = train_epoch(model, train_loader, optimizer, device, config['loss'])
            
            # 记录训练指标到wandb
            run.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/l1_error': train_metrics.get('l1_error', 0),
                'train/rmse': train_metrics.get('rmse', 0),
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            # 验证阶段
            if (epoch + 1) % config['training']['eval_every'] == 0:
                test_loss, test_metrics = evaluate(model, test_loader, device, config['loss'])
                
                print(f"📊 验证结果:")
                print(f"   Loss: {test_loss:.6f}")
                print(f"   L1: {test_metrics['l1_error']:.6f}")
                print(f"   RMSE: {test_metrics['rmse']:.6f}")
                print(f"   Last prediction: {test_metrics['last_prediction']}")
                print(f"   Last target: {test_metrics['last_target']}")
                
                # 记录验证指标到wandb
                run.log({
                    'epoch': epoch,
                    'val/loss': test_loss,
                    'val/l1_error': test_metrics['l1_error'],
                    'val/rmse': test_metrics['rmse']
                })
                
                # 学习率调度
                scheduler.step(test_loss)
                
                # 保存最佳模型
                if test_loss < best_loss:
                    best_loss = test_loss
                    patience_counter = 0
                    best_model_path = os.path.join(output_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': test_loss,
                        'metrics': test_metrics,
                        'config': config
                    }, best_model_path)
                    print(f"💾 保存最佳模型: {best_model_path}")
                    
                    # 保存模型到wandb
                    wandb.save(best_model_path)
                else:
                    patience_counter += 1
                
                # 早停检查
                if patience_counter >= config['training']['patience']:
                    print(f"⏰ 早停: {config['training']['patience']} 轮无改善")
                    break
            
            # 定期保存检查点
            if (epoch + 1) % config['training']['save_every'] == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    'config': config
                }, checkpoint_path)
                print(f"💾 保存检查点: {checkpoint_path}")
        
        print(f"🎉 训练完成! 最佳验证损失: {best_loss:.6f}")
        
        # 记录最终结果
        run.log({'final/best_loss': best_loss})
        
        return best_model_path
        
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {e}")
        raise
    finally:
        # 结束wandb运行
        run.finish()


def train_epoch(model, train_loader, optimizer, device, loss_config):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_metrics = {}
    total_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        # 获取批次数据
        forces_l = batch['forces_l'].to(device)  # (B, 3, 20, 20)
        forces_r = batch['forces_r'].to(device)  # (B, 3, 20, 20)
        actions = batch['action'].to(device)     # (B, 6) 但我们只需要前3维
        
        # 只使用位置增量 (dx, dy, dz)
        targets = actions[:, :3]  # (B, 3)
        
        batch_size = forces_l.size(0)
        if batch_size == 0:
            continue
        
        # 前向传播
        optimizer.zero_grad()
        predictions = model(forces_l, forces_r)  # (B, 3)
        
        # 计算损失
        loss, metrics = compute_feature_mlp_losses(predictions, targets, loss_config)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累加统计
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # 累加指标（跳过列表类型）
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value * batch_size
    
    # 计算平均值
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {key: value / max(total_samples, 1) for key, value in total_metrics.items()}
    
    print(f"🔄 训练: Loss={avg_loss:.6f}, L1={avg_metrics.get('l1_error', 0):.6f}, RMSE={avg_metrics.get('rmse', 0):.6f}")
    
    return avg_loss, avg_metrics


def evaluate(model, test_loader, device, loss_config):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_metrics = {}
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            forces_l = batch['forces_l'].to(device)
            forces_r = batch['forces_r'].to(device)
            actions = batch['action'].to(device)
            
            targets = actions[:, :3]  # 只使用位置增量
            batch_size = forces_l.size(0)
            
            if batch_size == 0:
                continue
            
            # 前向传播
            predictions = model(forces_l, forces_r)
            
            # 计算损失
            loss, metrics = compute_feature_mlp_losses(predictions, targets, loss_config)
            
            # 累加统计
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # 累加指标（跳过列表类型）
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in total_metrics:
                        total_metrics[key] = 0.0
                    total_metrics[key] += value * batch_size
                elif key in ['last_prediction', 'last_target']:
                    # 保留最后一组预测值和真实值
                    total_metrics[key] = value
    
    # 计算平均值
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {key: value / max(total_samples, 1) if isinstance(value, (int, float)) else value 
                   for key, value in total_metrics.items()}
    
    return avg_loss, avg_metrics


def main(config):
    """主函数"""
    print("🎯 Feature-MLP训练开始")
    print(f"📊 配置摘要:")
    print(f"   数据根目录: {config['data']['data_root']}")
    print(f"   批次大小: {config['data']['batch_size']}")
    print(f"   学习率: {config['training']['lr']}")
    print(f"   训练轮数: {config['training']['epochs']}")
    print(f"   输出目录: {config['output']['output_dir']}")
    
    best_model_path = train_feature_mlp(config)
    print(f"✅ 训练完成，最佳模型: {best_model_path}")
    
    return best_model_path


if __name__ == '__main__':
    # 默认配置 - 使用相对路径
    config = {
        'data': {
            'data_root': 'datasets/data25.7_aligned',  # 相对于project_root的路径
            'categories': [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ],
            'train_split': 0.8,
            'batch_size': 32,
            'num_workers': 4,
            'start_frame': 0,
            'normalize_config': {
                'forces': 'zscore',    # 触觉力数据标准化
                'actions': 'minmax',   # 动作数据归一化到[-1,1]
                'end_states': None,    # 不使用状态数据
                'resultants': None     # 不使用合力数据
            }
        },
        'model': {
            'feature_dim': 128,
            'action_dim': 3,  # 输出 (dx, dy, dz)
            'hidden_dims': [256, 128],  # 256 → 128 → 3
            'dropout_rate': 0.25,       # 提高到0.25
            'pretrained_encoder_path': 'tactile_representation/prototype_library/cnnae_crt_128.pt'  # 相对路径
        },
        'loss': {
            'loss_type': 'huber',
            'huber_delta': 1.0
        },
        'training': {
            'epochs': 100,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'eval_every': 5,
            'save_every': 20,
            'patience': 15
        },
        'output': {
            'output_dir': 'policy_learn/checkpoints'  # 相对路径，实际会在下面创建时间戳子文件夹
        },
        'wandb': {
            'project': 'tactile-feature-mlp',
            'tags': ['feature-mlp', 'tactile', 'behavior-cloning'],
            'notes': 'Feature-MLP training with pretrained tactile encoder and flexible dataset'
        }
    }
    
    # 检查路径
    data_path = os.path.join(project_root, config['data']['data_root'])
    pretrained_path = os.path.join(project_root, config['model']['pretrained_encoder_path'])
    
    print(f"🎯 Feature-MLP训练配置:")
    print(f"   项目根目录: {project_root}")
    print(f"   数据路径: {data_path}")
    print(f"   预训练权重: {pretrained_path}")
    print(f"   输出目录: {os.path.join(project_root, config['output']['output_dir'])}")
    
    if os.path.exists(data_path):
        print(f"✅ 数据路径存在")
    else:
        print(f"❌ 数据路径不存在: {data_path}")
    
    if os.path.exists(pretrained_path):
        print(f"✅ 预训练权重存在")
    else:
        print(f"⚠️  预训练权重不存在: {pretrained_path}")
        print("将使用随机初始化")
        config['model']['pretrained_encoder_path'] = ''
    
    # 开始训练
    print("\n" + "="*60)
    print("准备开始训练，请确保:")
    print("1. 已安装wandb: pip install wandb")
    print("2. 已登录wandb: wandb login")
    print("3. 数据路径正确")
    print("="*60)
    
    # 取消注释下面这行来实际运行训练
    main(config)
