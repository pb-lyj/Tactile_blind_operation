# Utils Package Documentation

## 概述

本utils包将原本散布在`Physical_mapping.py`和各个模型文件中的工具函数进行了重构和整理，提供了统一的接口和更好的模块化设计。

## 📁 项目结构

当前utils包包含以下模块：

```
utils/
├── __init__.py              # 统一导入接口
├── logging.py              # 日志记录功能  
├── config.py               # 配置管理功能
├── visualization.py        # 可视化工具
├── data_utils.py           # 数据处理工具
├── README.md               # 使用文档
└── demo_*.py               # 示例脚本
```

## 🆕 最新更新

### 改进版STN原型自编码器 (`ImprovedPrototypeSTNAE`)

新增了基于STN的改进原型自编码器，位于 `models/improved_prototype_ae_STN.py`：

**主要特性：**
- 🔧 **Xavier初始化**: 改进的原型和权重初始化
- 🎯 **STN空间变换**: 支持共享和独立两种STN模式  
- 🛡️ **随机扰动**: 训练时添加噪声防止过拟合
- 📊 **改进损失**: 包含STN正则化和基尼系数稀疏性损失
- 🏗️ **深度网络**: 3层CNN编码器，BatchNorm + Dropout

**快速使用：**
```python
from models.improved_prototype_ae_STN import ImprovedPrototypeSTNAE
model = ImprovedPrototypeSTNAE(num_prototypes=8, share_stn=True)
```

**相关文件：**
- `models/improved_prototype_ae_STN.py` - 模型实现
- `models/README_improved_STN.md` - 详细文档
- `test_improved_stn.py` - 功能测试
- `demo_improved_stn_training.py` - 训练演示

## 功能模块

### 1. 日志记录 (`utils.logging`)

提供同时输出到终端和文件的日志记录功能。

```python
from utils.logging import Logger
import sys

# 使用方法
sys.stdout = Logger("train.log", to_terminal=True, with_timestamp=True)
print("这条消息会同时输出到终端和日志文件")
```

**特性:**
- 同时输出到终端和文件
- 可选时间戳
- 自动文件关闭处理

### 2. 配置管理 (`utils.config`)

提供JSON格式的配置文件保存和加载功能。

```python
from utils.config import save_config_to_json, load_config_from_json

# 保存配置
config = {"model": {"num_prototypes": 8}, "training": {"lr": 0.001}}
save_config_to_json(config, "config.json", overwrite=True)

# 加载配置
config = load_config_from_json("config.json")
```

### 3. 可视化工具 (`utils.visualization`)

包含多种原型和激活序列的可视化功能。

#### 原型物理映射可视化
```python
from utils.visualization import save_physicalXYZ_images

# prototypes shape: (K, 3, H, W)
save_physicalXYZ_images(prototypes, output_dir="./prototype_images")
```

#### 激活序列热力图
```python
from utils.visualization import plot_activation_heatmap, plot_dual_activation_heatmap

# 单传感器激活热力图
plot_activation_heatmap(weights_sequence, "Activation Heatmap", "heatmap.png")

# 双传感器对比热力图
plot_dual_activation_heatmap(
    weights_left, weights_right, 
    "Dual Sensor Comparison", "dual_heatmap.png"
)
```

#### 原型使用分布
```python
from utils.visualization import plot_prototype_usage

plot_prototype_usage(weights, save_path="usage.png", title="Prototype Usage")
```

#### 训练损失曲线
```python
from utils.visualization import plot_loss_curves

# loss_history 可以是 list of dict 或 dict of list
plot_loss_curves(loss_history, save_path="losses.png")
```

### 4. 数据处理工具 (`utils.data_utils`)

提供数据保存、加载和统计分析功能。

#### 激活记录管理
```python
from utils.data_utils import save_activation_records, load_activation_records

# 保存激活记录
save_activation_records(activation_records, "records.pkl", format='pickle')

# 加载激活记录
records = load_activation_records("records.pkl")
```

#### 原型统计分析
```python
from utils.data_utils import compute_prototype_statistics

stats = compute_prototype_statistics(weights)
# 返回: 使用率、基尼系数、活跃原型数等统计信息
```

#### 样本权重分析
```python
from utils.data_utils import save_sample_weights_and_analysis

# 综合权重分析（包含t-SNE、分布图、统计报告等）
save_sample_weights_and_analysis(
    model=trained_model,
    dataset=test_dataset,
    output_dir="./analysis_results",
    max_samples_tsne=5000
)
```

**生成的文件:**
- `sample_weights.npy` - 权重数据
- `tsne_weights.png` - t-SNE可视化图
- `max_weight_hist.png` - 最大权重分布图  
- `prototype_usage.png` - 原型使用频率图
- `weight_analysis_report.txt` - 详细统计报告

#### 训练检查点
```python
from utils.data_utils import save_training_checkpoint, load_training_checkpoint

# 保存检查点
save_training_checkpoint(
    model.state_dict(), optimizer.state_dict(), 
    epoch, loss_history, "checkpoint.pt"
)

# 加载检查点
model_state, opt_state, epoch, history = load_training_checkpoint("checkpoint.pt")
```

## 迁移指南

### 原有引用替换

**之前:**
```python
from tactile_clustering.Physical_mapping import Logger, save_physicalXYZ_images
```

**现在:**
```python
from utils.logging import Logger
from utils.visualization import save_physicalXYZ_images
```

**或者统一导入:**
```python
from utils import Logger, save_physicalXYZ_images
```

### 已更新的文件

以下文件的引用已经更新：

1. `tactile_clustering/forces_prototype_discovery.py` - 更新Logger, save_physicalXYZ_images, save_sample_weights_and_analysis
2. `tactile_clustering/validate_prototype.py` - 更新Logger, save_plot_activation_sequences
3. `tactile_clustering/data_driven_prototype_discovery.py` - 更新Logger, save_physicalXYZ_images, save_sample_weights_and_analysis
4. `APT.py` - 更新Logger

### 新增功能

重构后新增了以下功能：

- `plot_prototype_usage()` - 原型使用分布图
- `plot_loss_curves()` - 训练损失曲线
- `compute_prototype_statistics()` - 原型使用统计
- `analyze_prototype_diversity()` - 原型多样性分析
- `save_training_checkpoint()` / `load_training_checkpoint()` - 训练检查点管理
- `save_sample_weights_and_analysis()` - **新增** 综合权重分析功能

## 示例代码

运行示例脚本查看所有功能：

```bash
python demo_utils.py
```

## 优势

1. **模块化设计**: 功能按用途分类，便于维护
2. **统一接口**: 提供统一的导入方式
3. **向后兼容**: 保持原有API不变
4. **功能增强**: 新增多项实用功能
5. **文档完善**: 每个函数都有详细文档说明

## 注意事项

1. 确保在项目根目录下运行，或者正确设置`sys.path`
2. 某些可视化功能需要matplotlib和seaborn
3. 数据分析功能可能需要sklearn
4. 保存大型激活记录时建议使用pickle格式
