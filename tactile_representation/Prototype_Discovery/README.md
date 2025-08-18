# Prototype Discovery Module

这个模块包含了重组后的原型发现系统，遵循模块化设计原则。

## 文件结构

```
tactile_representation/Prototype_Discovery/
├── datasets/                   # 数据集模块
│   ├── __init__.py
│   └── tactile_dataset.py     # 统一的触觉数据集类
├── training/                   # 训练模块
│   ├── __init__.py
│   ├── train_baseline.py      # 基础原型自编码器训练
│   ├── train_stn.py          # STN原型自编码器训练
│   ├── train_improved.py     # 改进版原型自编码器训练
│   └── train_improved_stn.py # 改进版STN原型自编码器训练
├── utils/                     # 工具模块
│   ├── __init__.py
│   ├── logging.py            # 日志工具
│   ├── visualization.py      # 可视化工具
│   └── data_utils.py         # 数据处理工具
└── main_train.py             # 主训练脚本
```

## 模型类型

1. **Baseline**: 基础原型自编码器 (`PrototypeAEBaseline`)
   - 参数量: ~15K
   - 简单的CNN编码器 + 原型重构

2. **STN**: 带空间变换网络的原型自编码器 (`PrototypeAutoencoder`)
   - 参数量: ~78K
   - 包含STN进行空间对齐

3. **Improved**: 改进版原型自编码器 (`ImprovedForcePrototypeAE`)
   - 参数量: ~117K
   - 更深的网络结构 + 多种正则化

4. **Improved STN**: 改进版STN原型自编码器 (`ImprovedPrototypeSTNAE`)
   - 参数量: ~238K
   - 结合STN和改进架构

## 使用方法

### 1. 单个模型训练

```bash
# 训练基础模型
python main_train.py --model baseline --epochs 50 --batch_size 64

# 训练STN模型
python main_train.py --model stn --epochs 50 --share_stn

# 训练改进版模型
python main_train.py --model improved --epochs 50

# 训练改进版STN模型
python main_train.py --model improved_stn --epochs 50
```

### 2. 批量训练所有模型

```bash
python main_train.py --model all --epochs 50
```

### 3. 自定义配置

```bash
python main_train.py --model baseline \
    --data_root ./data/data25.7_aligned \
    --num_prototypes 16 \
    --lr 1e-4 \
    --batch_size 128 \
    --epochs 100 \
    --diversity_lambda 0.1 \
    --entropy_lambda 10.0
```

### 4. 直接调用训练模块

```python
from training.train_baseline import main as train_baseline
from training.train_stn import main as train_stn

# 自定义配置
config = {
    'data': {...},
    'model': {...},
    'training': {...},
    'loss': {...},
    'output': {...}
}

# 训练模型
model, loss_history = train_baseline(config)
```

## 配置参数

### 数据配置
- `data_root`: 数据根目录
- `categories`: 数据类别列表
- `start_frame`: 起始帧
- `exclude_test_folders`: 是否排除测试文件夹
- `num_workers`: 数据加载器工作进程数

### 模型配置
- `num_prototypes`: 原型数量
- `input_shape`: 输入形状 (C, H, W)
- `feature_dim`: 特征维度（仅improved模型）
- `share_stn`: 是否共享STN（仅STN模型）

### 训练配置
- `batch_size`: 批大小
- `epochs`: 训练轮数
- `lr`: 学习率
- `weight_decay`: 权重衰减
- `patience`: 早停耐心值

### 损失配置
- `diversity_lambda`: 多样性损失权重
- `entropy_lambda`: 熵损失权重
- `sparsity_lambda`: 稀疏性损失权重
- `gini_lambda`: Gini损失权重
- `stn_reg_lambda`: STN正则化损失权重

## 输出文件

每次训练都会在 `./cluster/prototype_library/` 下创建时间戳目录，包含：

- `best_model.pt`: 最佳模型权重
- `final_model.pt`: 最终模型权重
- `prototypes.npy`: 学习到的原型
- `prototype_*.png`: 原型可视化图像
- `training_loss_curves.png`: 训练损失曲线
- `loss_history.npy`: 损失历史数据
- `sample_weights.npy`: 样本权重
- `prototype_usage.png`: 原型使用分析图
- `weight_analysis.txt`: 权重分析报告
- `train_*.log`: 训练日志

## 模块依赖

确保以下依赖已安装：
- torch
- numpy
- matplotlib
- tqdm
- scikit-learn (用于数据分析)

## 注意事项

1. 所有训练脚本都会自动创建输出目录和日志文件
2. 训练过程中会自动保存最佳模型和进行早停
3. 训练完成后会自动进行样本权重分析和可视化
4. 可以通过命令行参数或配置字典来控制训练行为
5. 所有模型都支持GPU训练（自动检测CUDA）

## 扩展

要添加新的模型类型：
1. 在 `training/` 目录下创建新的训练脚本
2. 在 `main_train.py` 中添加对应的训练函数
3. 更新配置和帮助信息
