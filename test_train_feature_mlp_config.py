"""
测试修改后的 Feature-MLP 训练脚本 (使用config字典)
"""

import os
import sys
import torch

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from policy_learn.training.train_feature_mlp import train_feature_mlp


def test_config_style_training():
    """测试使用config字典的训练方式"""
    print("=" * 60)
    print("测试 Feature-MLP 训练 (Config字典风格)")
    print("=" * 60)
    
    # 创建测试config字典 (类似train_cnn_autoencoder.py的风格)
    config = {
        'data': {
            'data_root': '/home/lyj/Program_python/Tactile_blind_operation/datasets/data25.7_aligned',
            'categories': ['cir_lar', 'cir_med'],  # 使用较少类别进行快速测试
            'train_split': 0.8,
            'batch_size': 2,  # 小批次用于测试
            'num_workers': 2,
            'start_frame': 0,
            'load_forces': True,
            'load_wrench': False,
            'load_end_effector': False
        },
        'model': {
            'feature_dim': 256,
            'action_dim': 3,
            'hidden_dims': [512, 512, 512],
            'dropout_rate': 0.1,
            'pretrained_encoder_path': None  # 将使用随机初始化
        },
        'loss': {
            'mse_weight': 1.0,
            'l1_weight': 0.1,
            'l2_weight': 0.001
        },
        'training': {
            'num_epochs': 2,  # 只训练2个epoch用于测试
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'scheduler_step': 10,
            'scheduler_gamma': 0.5,
            'eval_every': 1,
            'save_every': 1,
            'early_stopping_patience': 5
        },
        'output': {
            'output_dir': './test_checkpoints/feature_mlp_config_test'
        }
    }
    
    print("配置字典:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 40)
    print("开始训练...")
    print("=" * 40)
    
    try:
        # 调用训练函数 (类似train_cnn_autoencoder.py的调用方式)
        train_feature_mlp(config)
        print("\n✅ 训练成功完成!")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 运行测试
    success = test_config_style_training()
    
    if success:
        print("\n🎉 测试通过! 修改后的训练脚本工作正常.")
    else:
        print("\n❌ 测试失败.")
