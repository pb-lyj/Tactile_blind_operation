"""
统一验证接口
支持验证不同类型的模型
"""

import os
import sys
import argparse
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# 添加 Prototype_Discovery 目录到路径
prototype_discovery_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if prototype_discovery_path not in sys.path:
    sys.path.insert(0, prototype_discovery_path)

from validating.validate_cnn_autoencoder import main as validate_cnn_autoencoder
from validating.validate_prototype_cnn_ae import main as validate_prototype_cnn_ae


def get_base_config():
    """获取基础配置"""
    return {
        'data': {
            'data_root': '/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
            'categories': [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ],
            'start_frame': 0,
            'exclude_test_folders': False,
            'num_workers': 8,
            'normalize_method': 'quantile_zscore'
        },
        'validation': {
            'batch_size': 32,
            'save_samples_batches': 5,
            'num_visualization_samples': 16
        }
    }


def get_model_configs():
    """获取不同模型的配置"""
    base_config = get_base_config()
    
    configs = {
        'cnn_autoencoder': {
            **base_config,
            'data': {
                **base_config['data'],
                'categories': [
                    "cir_lar", "cir_med", "cir_sma"
                ]
            },
            'model': {
                'in_channels': 3,
                'latent_dim': 256,
                'model_path': '../prototype_library/2025-08-17_17-56-57_cnn_autoencoder/final_model.pt'
            },
            'loss': {
                'l2_lambda': 0.001
            },
            'output': {
                'output_dir': os.path.join("./validation_results", 
                                         datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_cnn_autoencoder")
            }
        },
        
        'prototype_cnn': {
            **base_config,
            'model': {
                'num_prototypes': 8,
                'input_shape': [3, 20, 20],
                'model_path': None
            },
            'loss': {
                'diversity_lambda': 1.0,
                'entropy_lambda': 0.1,
                'sparsity_lambda': 0.01
            },
            'output': {
                'output_dir': os.path.join("./validation_results", 
                                         datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_prototype_cnn_ae")
            }
        }
    }
    
    return configs


def main():
    """主验证函数"""
    parser = argparse.ArgumentParser(description='统一验证接口')
    parser.add_argument('--model', type=str, required=True,
                       choices=['cnn_autoencoder', 'prototype_cnn'],
                       help='选择要验证的模型类型')
    parser.add_argument('--model_path', type=str, default=None,
                       help='指定模型路径，如果不指定则自动寻找最新模型')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='验证批次大小')
    parser.add_argument('--categories', nargs='+', default=None,
                       help='指定验证的数据类别')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='指定输出目录')
    
    args = parser.parse_args()
    
    # 获取配置
    configs = get_model_configs()
    config = configs[args.model].copy()
    
    # 应用命令行参数
    if args.model_path:
        config['model']['model_path'] = args.model_path
    
    if args.batch_size:
        config['validation']['batch_size'] = args.batch_size
    
    if args.categories:
        config['data']['categories'] = args.categories
    
    if args.output_dir:
        config['output']['output_dir'] = args.output_dir
    
    # 打印配置信息
    print("=" * 60)
    print(f"验证模型: {args.model}")
    print(f"数据类别: {config['data']['categories']}")
    print(f"批次大小: {config['validation']['batch_size']}")
    print(f"输出目录: {config['output']['output_dir']}")
    if config['model'].get('model_path'):
        print(f"模型路径: {config['model']['model_path']}")
    else:
        print("❌ 错误：未指定模型路径")
        print("使用 --model_path 参数指定模型文件路径")
        return
    print("=" * 60)
    
    # 选择对应的验证函数
    if args.model == 'cnn_autoencoder':
        print("🚀 开始验证CNN AutoEncoder...")
        model, results = validate_cnn_autoencoder(config)
    elif args.model == 'prototype_cnn':
        print("🚀 开始验证Prototype CNN AutoEncoder...")
        model, results = validate_prototype_cnn_ae(config)
    else:
        print(f"❌ 不支持的模型类型: {args.model}")
        return
    
    if model is not None and results is not None:
        print("✅ 验证完成！")
        print(f"📁 结果保存在: {config['output']['output_dir']}")
    else:
        print("❌ 验证失败！")


if __name__ == '__main__':
    main()
