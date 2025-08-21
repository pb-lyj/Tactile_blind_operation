# Feature-MLP训练系统使用指南

## 🎯 概述

本系统实现了基于预训练触觉特征的Feature-MLP模型训练，使用wandb进行实验跟踪和可视化。

## 📁 文件结构

```
├── policy_learn/
│   ├── models/
│   │   └── feature_mlp_new.py          # Feature-MLP模型定义
│   ├── training/
│   │   └── train_feature_mlp_new.py    # 训练脚本（支持wandb）
│   └── dataset_dataloader/
│       └── flexible_policy_dataset.py  # 灵活数据集加载器
├── run_feature_mlp_training.py         # 完整训练启动脚本
├── run_test_training.py                # 小规模测试训练脚本  
├── test_wandb_login.py                 # wandb登录测试脚本
└── test_basic_functionality.py        # 基本功能测试（无wandb）
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活conda环境
conda activate data_ans

# 安装wandb（如果还没安装）
pip install wandb

# 登录wandb
wandb login
```

### 2. 测试wandb配置

```bash
python test_wandb_login.py
```

### 3. 运行基本功能测试

```bash
python test_basic_functionality.py
```

### 4. 运行小规模测试训练

```bash
python run_test_training.py
```

### 5. 运行完整训练

```bash
python run_feature_mlp_training.py
```

## ⚙️ 配置说明

训练配置包含以下主要部分：

### 数据配置
```python
'data': {
    'data_root': 'datasets/data25.7_aligned',  # 数据集路径（相对于项目根目录）
    'categories': [...],                       # 要训练的数据类别
    'batch_size': 32,                         # 批次大小
    'normalize_config': {                     # 归一化配置
        'forces': 'zscore',                   # 触觉力数据标准化
        'actions': 'minmax'                   # 动作数据归一化到[-1,1]
    }
}
```

### 模型配置
```python
'model': {
    'feature_dim': 128,                                           # 单手特征维度
    'action_dim': 3,                                             # 输出动作维度 (dx, dy, dz)
    'hidden_dims': [512, 512, 512],                             # MLP隐藏层
    'pretrained_encoder_path': 'tactile_representation/...'      # 预训练权重路径
}
```

### 训练配置
```python
'training': {
    'epochs': 100,        # 最大训练轮数
    'lr': 1e-4,          # 学习率
    'eval_every': 5,     # 验证频率
    'patience': 15       # 早停耐心值
}
```

### wandb配置
```python
'wandb': {
    'project': 'tactile-feature-mlp',                    # wandb项目名
    'entity': 'your-team-name',                          # 您的wandb用户名或团队名
    'tags': ['feature-mlp', 'tactile', 'behavior-cloning'] # 标签
}
```

## 📊 训练输出

### 文件输出
训练会在`policy_learn/checkpoints/`下创建时间戳命名的文件夹：
```
policy_learn/checkpoints/feature_mlp_20250821_143022/
├── best_model.pt              # 最佳模型权重
├── checkpoint_epoch_20.pt     # 定期检查点
├── checkpoint_epoch_40.pt
└── wandb/                     # wandb本地文件
```

### wandb仪表板
- **训练指标**: train/loss, train/l1_error, train/rmse
- **验证指标**: val/loss, val/l1_error, val/rmse
- **学习率**: learning_rate
- **模型文件**: 自动上传最佳模型

## 🎛️ 模型架构

```
输入: 左右手触觉力数据 (B, 3, 20, 20) × 2
  ↓
预训练CNN编码器 (冻结)
  ↓
特征提取: 128维 × 2 = 256维
  ↓
MLP网络: 256 → 512 → 512 → 512 → 3
  ↓
输出: 3维位置增量 (dx, dy, dz)
```

## 🔧 自定义配置

### 修改数据类别
```python
'categories': ["cir_lar", "rect_med"]  # 只训练特定类别
```

### 调整网络结构
```python
'hidden_dims': [256, 128, 64]  # 更小的网络
```

### 更改loss函数
```python
'loss': {
    'loss_type': 'mse',    # 可选: 'huber', 'mse', 'l1'
    'huber_delta': 1.0     # Huber loss参数
}
```

## 🐛 常见问题

### wandb登录问题
```bash
# 重新登录
wandb login --relogin

# 检查登录状态
wandb status
```

### CUDA内存不足
```python
# 减小批次大小
'batch_size': 16  # 或更小

# 减少工作进程
'num_workers': 2
```

### 数据路径错误
确保数据路径相对于项目根目录正确：
```python
'data_root': 'datasets/data25.7_aligned'  # 不要用绝对路径
```

## 📈 监控训练

1. **wandb仪表板**: 访问 https://wandb.ai 查看实时训练图表
2. **终端输出**: 实时查看训练进度和指标
3. **本地文件**: 检查保存的模型和检查点

## 💡 最佳实践

1. **先运行测试**: 使用`run_test_training.py`验证配置
2. **监控过拟合**: 关注训练/验证损失差距
3. **调整学习率**: 根据损失曲线调整学习率计划
4. **保存实验**: 为每次实验添加有意义的wandb标签和备注

## 🎉 开始训练

确保完成环境配置后，直接运行：
```bash
python run_feature_mlp_training.py
```

系统会自动处理：
- ✅ wandb登录和初始化
- ✅ 数据集加载和预处理  
- ✅ 模型创建和预训练权重加载
- ✅ 训练循环和验证
- ✅ 最佳模型保存
- ✅ 实验跟踪和可视化
