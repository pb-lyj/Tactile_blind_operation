"""
主训练脚本 - 统一管理所有原型自编码器模型训练
"""

import os
import sys
import torch
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from training.train_prototype_cnn_ae import main as train_prototype_cnn_ae_main
from training.train_prototype_stn_ae import main as train_prototype_stn_ae_main
from training.train_cnn_autoencoder import main as train_cnn_autoencoder_main
from training.train_vqgan import main as train_vqgan_main


def get_base_config():
    """
    获取基础配置
    """
    return {
        'data': {
            'data_root': '/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
            'categories': [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma",
                "tri_lar", "tri_med", "tri_sma"
            ],
            'start_frame': 0,
            'exclude_test_folders': True,
            'normalize_method' : 'quantile_zscore',
            'num_workers': 8
        },
        'model': {
            'input_shape': (3, 20, 20),
            'feature_dim': 128,
            'share_stn': True
        },
        'training': {
            'batch_size': 64,
            'epochs': 10,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'patience': 20
        },
        'loss': {
            'diversity_lambda': 0.1,
            'entropy_lambda': 5.0,
            'sparsity_lambda': 0.01,
            'gini_lambda': 0.05,
            'stn_reg_lambda': 0.05
        }
    }


def get_model_configs():
    """
    获取不同模型的配置
    """
    base_config = get_base_config()
    
    configs = {}
    
    
    # 原型自编码器
    configs['prototype_cnn'] = base_config.copy()
    configs['prototype_cnn']['model']['num_prototypes'] = 8
    configs['prototype_cnn']['output'] = {
        'output_dir': os.path.join("../prototype_library", 
                                 datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_prototype_cnn")
    }
    
    # STN原型自编码器
    configs['prototype_stn'] = base_config.copy()
    configs['prototype_cnn']['model']['num_prototypes'] = 8
    configs['prototype_stn']['output'] = {
        'output_dir': os.path.join("../prototype_library", 
                                 datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_prototype_stn")
    }
    
    # CNN自编码器
    configs['cnn_autoencoder'] = {
        'data': base_config['data'].copy(),
        'model': {
            'in_channels': 3,
            'latent_dim': 128
        },
        'loss': {
            'l2_lambda': 0.001
        },
        'training': {
            'batch_size': 64,
            'epochs': 50,
            'lr': 1e-4,
            'weight_decay': 1e-4
        },
        'output': {
            'output_dir': os.path.join("../prototype_library", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_cnn_autoencoder")
        }
    }
    
    
    # VQGAN
    configs['vqgan'] = {
        'data': base_config['data'].copy(),
        'model': {
            'in_channels': 3,
            'latent_dim': 128,
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
            'epochs': 50,
            'lr': 1e-4,
            'disc_lr': 1e-4,
            'weight_decay': 1e-5
        },
        'output': {
            'output_dir': os.path.join("../prototype_library", 
                                     datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_vqgan")
        }
    }
    
    
    return configs


def train_model(model_type, config=None, **kwargs):
    """
    训练指定类型的模型
    Args:
        model_type: 模型类型 
        config: 自定义配置（可选）
        **kwargs: 其他配置参数
    """
    # 获取默认配置
    configs = get_model_configs()
    
    # 更新配置
    if config is not None:
        if model_type in configs:
            configs[model_type].update(config)
    
    # 从kwargs更新配置
    for key, value in kwargs.items():
        if '.' in key:
            # 处理嵌套键，如 'training.epochs'
            section, param = key.split('.', 1)
            if model_type in configs and section in configs[model_type]:
                configs[model_type][section][param] = value
        else:
            # 顶级参数
            if model_type in configs and key in configs[model_type]:
                configs[model_type][key] = value

    
    if model_type == 'prototype_cnn':
        print("🚀 开始训练CNN原型自编码器...")
        return train_prototype_cnn_ae_main(configs['prototype_cnn'])
    
    elif model_type == 'prototype_stn':
        print("🚀 开始训练STN原型自编码器...")
        return train_prototype_stn_ae_main(configs['prototype_stn'])
    
    elif model_type == 'cnn_autoencoder':
        print("🚀 开始训练CNN自编码器...")
        return train_cnn_autoencoder_main(configs['cnn_autoencoder'])
    
    elif model_type == 'vqgan':
        print("🚀 开始训练VQGAN...")
        return train_vqgan_main(configs['vqgan'])
    
    elif model_type == 'all':
        print("🚀 开始训练所有模型...")
        results = {}
        
        for model_name in ['prototype_cnn', 'prototype_stn', 'cnn_autoencoder', 'vqgan']:
            print(f"\n{'='*60}")
            print(f"训练模型: {model_name}")
            print(f"{'='*60}")
            
            try:
                results[model_name] = train_model(model_name, configs[model_name])
                print(f"✅ {model_name} 训练成功完成!")
            except Exception as e:
                print(f"❌ {model_name} 训练失败: {str(e)}")
                results[model_name] = None
        
        return results
    
    else:
        raise ValueError(f"未知的模型类型: {model_type}. 支持的类型: ['prototype_cnn', 'stprototype_stnn', 'cnn_autoencoder', 'vqgan', 'all']")


def main():
    """
    主函数 - 命令行接口
    命令行中接收的参数将覆盖配置文件中的默认值
    参数优先级：
        命令行参数(cmd)  >   配置文件中的默认值(train_main)  >   函数内部的硬编码默认值(train)
    """
    parser = argparse.ArgumentParser(description="原型自编码器训练脚本")
    
    # 基本参数
    parser.add_argument('--model', type=str, default='prototype_cnn',
                       choices=['prototype_cnn', 'prototype_stn', 'cnn_autoencoder', 'vqgan', 'all'],
                       help='要训练的模型类型')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
                       help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='数据加载器工作进程数')
    
    # 模型参数
    parser.add_argument('--num_prototypes', type=int, default=8,
                       help='原型数量')
    parser.add_argument('--feature_dim', type=int, default=128,
                       help='特征维度（仅用于feature模型）')
    parser.add_argument('--share_stn', action='store_true', default=True,
                       help='是否共享STN（仅用于STN模型）')
    
    # 损失参数
    parser.add_argument('--diversity_lambda', type=float, default=0.1,
                       help='多样性损失权重')
    parser.add_argument('--entropy_lambda', type=float, default=10.0,
                       help='熵损失权重')
    parser.add_argument('--sparsity_lambda', type=float, default=0.01,
                       help='稀疏性损失权重')
    parser.add_argument('--gini_lambda', type=float, default=0.05,
                       help='Gini损失权重')
    parser.add_argument('--stn_reg_lambda', type=float, default=0.05,
                    help='STN正则化损失权重')
    
    # CNN Autoencoder参数
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='潜在空间维度（用于CNN autoencoder和VQGAN）')
    
    # VQGAN参数
    parser.add_argument('--num_embeddings', type=int, default=1024,
                       help='码本大小（用于VQGAN）')
    parser.add_argument('--commitment_cost', type=float, default=0.25,
                       help='承诺损失权重（用于VQGAN）')
    parser.add_argument('--disc_lr', type=float, default=1e-4,
                       help='判别器学习率（用于VQGAN）')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--patience', type=int, default=20,
                       help='早停耐心值')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录（如果不指定，将自动生成）')
    
    args = parser.parse_args()  # 解析命令行参数，  Namespace 对象 使用 . 访问
    
    # 构建配置更新
    config_updates = {}
    
    # 加载命令行配置进配置字典
    # 数据配置
    config_updates['data.data_root'] = args.data_root
    config_updates['training.batch_size'] = args.batch_size
    config_updates['data.num_workers'] = args.num_workers
    
    # 模型配置
    config_updates['model.num_prototypes'] = args.num_prototypes
    config_updates['model.feature_dim'] = args.feature_dim
    config_updates['model.share_stn'] = args.share_stn
    config_updates['model.latent_dim'] = args.latent_dim
    config_updates['model.num_embeddings'] = args.num_embeddings
    config_updates['model.commitment_cost'] = args.commitment_cost
    
    # 训练配置
    config_updates['training.epochs'] = args.epochs
    config_updates['training.lr'] = args.lr
    config_updates['training.disc_lr'] = args.disc_lr
    config_updates['training.weight_decay'] = args.weight_decay
    config_updates['training.patience'] = args.patience
    
    # 损失配置
    config_updates['loss.diversity_lambda'] = args.diversity_lambda
    config_updates['loss.entropy_lambda'] = args.entropy_lambda
    config_updates['loss.sparsity_lambda'] = args.sparsity_lambda
    config_updates['loss.gini_lambda'] = args.gini_lambda
    config_updates['loss.stn_reg_lambda'] = args.stn_reg_lambda
    
    # 输出配置
    if args.output_dir is not None:
        config_updates['output.output_dir'] = args.output_dir
    
    # 开始训练
    print("🎯 原型自编码器训练系统")
    print(f"模型类型: {args.model}")
    print(f"数据根目录: {args.data_root}")
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print("-" * 60)
    
    try:
        result = train_model(args.model, None, **config_updates)
        if args.model == 'all':
            print("\n🎉 所有模型训练完成!")
            for model_name, model_result in result.items():
                status = "✅ 成功" if model_result is not None else "❌ 失败"
                print(f"  {model_name}: {status}")
        else:
            print(f"\n🎉 {args.model} 模型训练完成!")
    
    except Exception as e:
        print(f"\n❌ 训练失败: {str(e)}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
