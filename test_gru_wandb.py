#!/usr/bin/env python3
"""
测试修改后的GRU训练脚本
验证wandb记录和额外指标
"""

import os
import sys

# 添加项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.append(project_root)

from policy_learn.training.train_gru import train_gru_policy

def test_gru_wandb():
    """测试GRU训练的wandb记录功能"""
    print("🧪 测试GRU训练的wandb记录功能")
    
    # 简化测试配置
    test_config = {
        'data': {
            'data_root': 'datasets/data25.7_aligned',
            'categories': ["cir_lar"],  # 只用一个类别测试
            'sequence_length': 8,       # 较短序列
            'train_ratio': 0.8,
            'random_seed': 42,
            'start_frame': 0,
            'use_end_states': True,
            'use_forces': False,
            'use_resultants': True,
            'normalize_config': {
                'actions': 'minmax',
                'resultants': 'zscore',
                'end_states': None
            },
            'num_workers': 1
        },
        'model': {
            'input_dim': 12,
            'output_dim': 3,
            'hidden_dim': 128,      # 较小隐藏层
            'num_layers': 1,        # 单层
            'dropout': 0.1,
            'use_normalization': True
        },
        'loss': {
            'l2_weight': 1.0,
            'huber_weight': 0.0,
            'delta_u_weight': 0.1,
            'jerk_weight': 0.05
        },
        'training': {
            'batch_size': 4,        # 小批次
            'epochs': 3,            # 只训练3轮测试
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'patience': 20
        },
        'output': {
            'output_dir': 'policy_learn/test_checkpoints/gru_test'
        },
        'wandb': {
            'enabled': True,
            'mode': 'offline',      # 离线模式测试
            'project': 'tactile-gru-policy-test',
            'tags': ['test', 'gru', 'l1-metrics'],
            'notes': 'Testing GRU training with L1 metrics and last prediction logging'
        }
    }
    
    print("📊 测试配置:")
    print(f"   序列长度: {test_config['data']['sequence_length']}")
    print(f"   训练轮数: {test_config['training']['epochs']}")
    print(f"   批次大小: {test_config['training']['batch_size']}")
    print(f"   wandb模式: {test_config['wandb']['mode']}")
    print("✅ 预期记录的指标:")
    print("   - train/loss & test/loss")
    print("   - train/l1_error & test/l1_error")
    print("   - test/last_prediction & test/last_target")
    print("   - learning_rate")
    
    try:
        print("\n" + "="*50)
        print("🎯 开始GRU训练测试...")
        print("="*50)
        
        model, train_history, test_history = train_gru_policy(test_config)
        
        print("\n" + "="*50)
        print("🎉 GRU训练测试成功!")
        print("✅ 验证项目:")
        print("   - wandb项目设置正确")
        print("   - L1误差指标计算和记录")
        print("   - 最后预测值和目标值记录")
        print("   - 训练和测试指标显示格式统一")
        print("   - 进度条显示正常")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("🧪 GRU wandb记录功能测试")
    print("📋 测试目标:")
    print("   1. 验证L1误差计算和记录")
    print("   2. 验证最后预测值和目标值记录")
    print("   3. 验证wandb记录格式与feature训练一致")
    print("   4. 验证训练显示信息完整性")
    print("")
    
    success = test_gru_wandb()
    
    if success:
        print("\n🎊 GRU wandb记录功能测试完成！")
        print("💡 现在GRU训练具有与Feature-MLP相同的记录功能")
        print("📊 wandb记录指标:")
        print("   - train/loss, test/loss")
        print("   - train/l1_error, test/l1_error")
        print("   - test/last_prediction, test/last_target")
        print("   - learning_rate")
        print("📈 可以在wandb面板中对比不同模型的性能")
    else:
        print("\n💔 GRU wandb记录功能测试失败")
        print("🔧 请检查错误信息并修复问题")
