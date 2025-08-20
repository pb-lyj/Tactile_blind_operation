# Feature-MLP 模型使用说明

## 概述

Feature-MLP 是一个基于预训练触觉特征的行为克隆模型，用于学习从触觉传感器数据到机器人动作增量的映射。

## 模型架构

```
输入: 
  - 左手触觉数据: (B, 3, 20, 20)
  - 右手触觉数据: (B, 3, 20, 20)

特征提取:
  - 预训练CNN自编码器 (冻结参数)
  - 左手特征: (B, 128)
  - 右手特征: (B, 128)
  - 连接特征: (B, 256)

MLP网络:
  - 输入维度: 256
  - 隐藏层: [512, 512, 512] + LayerNorm + ReLU + Dropout
  - 输出维度: 3 (delta_x, delta_y, delta_z)

输出:
  - 动作增量: (B, 3)
```

## 文件结构

```
policy_learn/
├── models/
│   └── feature_mlp.py              # 模型定义
├── dataset_dataloader/
│   └── policy_dataset.py           # 数据集类
└── training/
    ├── train_feature_mlp.py        # 训练脚本
    ├── configs/
    │   └── feature_mlp_config.json # 配置文件
    └── TRAINING.md                 # 训练说明
```

## 快速开始

### 1. 准备数据

确保数据位于正确路径：
```
datasets/data25.7_aligned/
├── cir_lar/
├── cir_med/
├── rect_lar/
└── rect_med/
```

每个数据目录应包含：
- `_action.npy`: 动作数据 (T, 7) - [timestamp, x, y, z, rx, ry, rz]
- `_forces_l.npy`: 左手触觉数据 (T, 3, 20, 20)
- `_forces_r.npy`: 右手触觉数据 (T, 3, 20, 20)

### 2. 配置预训练权重

修改配置文件中的预训练权重路径：
```json
{
  "model": {
    "pretrained_encoder_path": "/path/to/your/tactile_encoder.pt"
  }
}
```

### 3. 训练模型

```bash
cd /home/lyj/Program_python/Tactile_blind_operation/policy_learn/training

# 使用默认配置
python train_feature_mlp.py

# 使用自定义配置
python train_feature_mlp.py --config configs/feature_mlp_config.json

# 指定保存目录
python train_feature_mlp.py --save_dir ./my_checkpoints

# 恢复训练
python train_feature_mlp.py --resume ./checkpoints/feature_mlp_20241220_123456/best_model.pt
```

### 4. 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir ./checkpoints/feature_mlp_*/tensorboard
```

## 配置参数

### 模型配置
- `feature_dim`: 单手特征维度 (默认: 128)
- `action_dim`: 输出动作维度 (默认: 3)
- `hidden_dims`: MLP隐藏层维度列表 (默认: [512, 512, 512])
- `dropout_rate`: Dropout率 (默认: 0.1)
- `pretrained_encoder_path`: 预训练编码器路径

### 数据配置
- `data_root`: 数据根目录
- `categories`: 包含的类别列表
- `train_ratio`: 训练集比例 (默认: 0.8)
- `random_seed`: 随机种子 (默认: 42)

### 训练配置
- `batch_size`: 批大小 (默认: 16)
- `learning_rate`: 学习率 (默认: 1e-4)
- `num_epochs`: 训练轮数 (默认: 100)
- `early_stopping_patience`: 早停耐心值 (默认: 20)

### 损失配置
- `loss_type`: 损失类型 (`huber`/`mse`/`l1`, 默认: `huber`)
- `huber_delta`: Huber损失的delta参数 (默认: 1.0)

## 模型使用

### 加载预训练模型

```python
import torch
from policy_learn.models.feature_mlp import FeatureMLP

# 加载模型
checkpoint = torch.load('path/to/best_model.pt')
config = checkpoint['config']

model = FeatureMLP(
    feature_dim=config['model']['feature_dim'],
    action_dim=config['model']['action_dim'],
    hidden_dims=config['model']['hidden_dims'],
    dropout_rate=config['model']['dropout_rate'],
    pretrained_encoder_path=config['model']['pretrained_encoder_path']
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### 推理

```python
# 准备输入数据
forces_l = torch.randn(1, 3, 20, 20)  # 左手触觉数据
forces_r = torch.randn(1, 3, 20, 20)  # 右手触觉数据

# 预测动作增量
with torch.no_grad():
    delta_action = model(forces_l, forces_r)
    print(f"预测的动作增量: {delta_action}")

# 详细预测信息
results = model.predict(forces_l, forces_r)
print(f"左手特征: {results['feature_l'].shape}")
print(f"右手特征: {results['feature_r'].shape}")
print(f"连接特征: {results['features'].shape}")
print(f"动作增量: {results['delta_action']}")
```

## 输出文件

训练过程会生成以下文件：

```
checkpoints/feature_mlp_20241220_123456/
├── config.json                    # 训练配置
├── training.log                   # 训练日志
├── best_model.pt                  # 最佳模型
├── final_model.pt                 # 最终模型
├── checkpoint_epoch_*.pt          # 定期检查点
└── tensorboard/                   # TensorBoard日志
    ├── events.out.tfevents.*
    └── ...
```

## 性能指标

训练过程中监控的指标：
- `total_loss`: 总损失
- `main_loss`: 主损失 (Huber/MSE/L1)
- `mae`: 平均绝对误差
- `mse`: 均方误差
- `rmse`: 均方根误差
- `error_x/y/z`: 各轴误差

## 故障排除

### 常见问题

1. **找不到预训练权重**
   - 检查 `pretrained_encoder_path` 是否正确
   - 可以设置为 `null` 使用随机初始化进行测试

2. **数据加载失败**
   - 检查数据路径和文件格式
   - 确认每个目录包含必要的 `.npy` 文件

3. **内存不足**
   - 减小 `batch_size`
   - 减少 `hidden_dims` 的大小

4. **训练不收敛**
   - 调整学习率
   - 检查数据质量
   - 尝试不同的损失函数

### 调试模式

运行测试脚本验证系统：
```bash
python test_feature_mlp_system.py
```

这将测试：
- 模型创建和前向传播
- 数据加载
- 训练步骤
- 损失计算

## 实验建议

1. **基线实验**: 使用少量类别 (如 `["cir_lar", "cir_med"]`) 进行快速验证
2. **消融研究**: 比较不同的 MLP 架构和损失函数
3. **数据扩展**: 逐步增加更多形状类别
4. **特征分析**: 可视化提取的触觉特征

## 下一步

- 实现 Feature-GRU 模型用于时序建模
- 集成到 ACT 和 Diffusion Policy 框架
- 添加数据增强和正则化技术
- 实现在线推理和实时控制接口
