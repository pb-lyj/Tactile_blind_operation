"""
Comparison模块统一训练入口 - 与Prototype Discovery保持一致的接口
支持VQ-VAE、MAE、BYOL三种方法
"""

import os
import sys
import argparse
import json
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "..")
sys.path.insert(0, project_root)

from training.train_vqvae import main as train_vqvae
from training.train_mae import main as train_mae
from training.train_byol import main as train_byol


def get_base_config():
    """
    获取基础配置 - 与Prototype Discovery保持一致
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
            'num_workers': 8
        },
        'model': {
            'input_shape': (3, 20, 20)
        },
        'training': {
            'batch_size': 64,
            'epochs': 50,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'patience': 20
        }
    }


def get_model_configs():
    """
    获取不同模型的配置
    """
    base_config = get_base_config()
    configs = {}
    
    # VQ-VAE配置
    configs['vqvae'] = base_config.copy()
    configs['vqvae']['model'].update({
        'num_embeddings': 512,
        'embedding_dim': 256,
        'num_hiddens': 256,
        'num_residual_layers': 4,
        'num_residual_hiddens': 256,
        'commitment_cost': 0.25
    })
    configs['vqvae']['output'] = {
        'output_dir': os.path.join("../comparison_library", 
                                 datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_vqvae")
    }
    
    # MAE配置
    configs['mae'] = base_config.copy()
    configs['mae']['model'].update({
        'patch_size': 4,
        'embed_dim': 256,
        'encoder_depth': 6,
        'decoder_depth': 4,
        'encoder_heads': 8,
        'decoder_heads': 8,
        'masking_ratio': 0.75
    })
    configs['mae']['output'] = {
        'output_dir': os.path.join("../comparison_library", 
                                 datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_mae")
    }
    
    # BYOL配置
    configs['byol'] = base_config.copy()
    configs['byol']['model'].update({
        'backbone': 'resnet18',
        'projection_dim': 128,
        'hidden_dim': 512,
        'temperature': 0.2,
        'tau': 0.996
    })
    configs['byol']['output'] = {
        'output_dir': os.path.join("../comparison_library", 
                                 datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_byol")
    }
    
    return configs


def train_model(model_type, config=None, **kwargs):
    """
    训练指定类型的模型 - 与Prototype Discovery保持一致的接口
    Args:
        model_type: 模型类型 ['vqvae', 'mae', 'byol', 'all']
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

    # 训练模型
    if model_type == 'vqvae':
        print("🚀 开始训练VQ-VAE模型...")
        return train_vqvae(configs['vqvae'])
    
    elif model_type == 'mae':
        print("🚀 开始训练MAE模型...")
        return train_mae(configs['mae'])
    
    elif model_type == 'byol':
        print("🚀 开始训练BYOL模型...")
        return train_byol(configs['byol'])
    
    elif model_type == 'all':
        print("🚀 开始训练所有模型...")
        results = {}
        
        for model_name in ['vqvae', 'mae', 'byol']:
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
        raise ValueError(f"未知的模型类型: {model_type}. 支持的类型: ['vqvae', 'mae', 'byol', 'all']")
def main():
    """
    主函数 - 命令行接口，与Prototype Discovery保持一致
    """
    parser = argparse.ArgumentParser(description="Comparison模块训练脚本")
    
    # 基本参数
    parser.add_argument('--model', type=str, default='vqvae',
                       choices=['vqvae', 'mae', 'byol', 'all'],
                       help='要训练的模型类型')
    
    # 数据参数
    parser.add_argument('--data_root', type=str, default='/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
                       help='数据根目录')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--num_workers', type=int, default=8,
                       help='数据加载器工作进程数')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
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
    
    args = parser.parse_args()
    
    # 构建配置
    config_updates = {}
    
    # 数据配置
    config_updates['data.data_root'] = args.data_root
    config_updates['training.batch_size'] = args.batch_size
    config_updates['data.num_workers'] = args.num_workers
    
    # 训练配置
    config_updates['training.epochs'] = args.epochs
    config_updates['training.lr'] = args.lr
    config_updates['training.weight_decay'] = args.weight_decay
    config_updates['training.patience'] = args.patience
    
    # 输出配置
    if args.output_dir is not None:
        config_updates['output.output_dir'] = args.output_dir
    
    # 开始训练
    print("🎯 Comparison模块训练系统")
    print(f"模型类型: {args.model}")
    print(f"数据根目录: {args.data_root}")
    print(f"训练轮数: {args.epochs}")
    print(f"批大小: {args.batch_size}")
    print("-" * 60)
    
    try:
        result = train_model(args.model, **config_updates)
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
