# 新增可视化功能说明

## 概述

在 `utils/visualization.py` 中新增了三个重要的原型分析可视化函数，用于深入分析原型自编码器的学习效果。

## 新增函数

### 1. `plot_maximum_weight_distribution(weights, save_path=None, title="Maximum Weight Distribution")`

**功能**: 绘制最大权重分布直方图
- 分析每个样本的最大激活权重分布
- 帮助理解模型对样本的专门化程度
- 显示权重分布的统计信息（均值、标准差等）

**输入参数**:
- `weights`: 权重数组，形状为(N, K)，N为样本数，K为原型数
- `save_path`: 保存路径（可选）
- `title`: 图表标题

### 2. `plot_average_prototype_usage(weights, save_path=None, title="Average Prototype Usage")`

**功能**: 绘制平均原型使用情况柱状图
- 显示每个原型的平均激活水平
- 识别过度使用或未充分使用的原型
- 评估原型库的平衡性

**输入参数**:
- `weights`: 权重数组，形状为(N, K)
- `save_path`: 保存路径（可选）
- `title`: 图表标题

### 3. `plot_tsne_sample_weights(weights, save_path=None, title="t-SNE of Sample Weights", n_samples=5000)`

**功能**: 绘制样本权重的t-SNE可视化
- 将高维权重空间映射到2D空间
- 揭示样本在原型空间中的聚类结构
- 帮助理解原型之间的关系

**输入参数**:
- `weights`: 权重数组，形状为(N, K)
- `save_path`: 保存路径（可选）
- `title`: 图表标题
- `n_samples`: 用于t-SNE的样本数量（默认5000，用于计算效率）

### 4. `create_comprehensive_prototype_analysis(weights, save_dir, prefix="prototype_analysis")`

**功能**: 创建全面的原型分析可视化
- 一次性生成所有四种可视化图表
- 包括：最大权重分布、原型使用情况、t-SNE可视化、权重分布直方图

**输入参数**:
- `weights`: 权重数组，形状为(N, K)
- `save_dir`: 保存目录
- `prefix`: 文件名前缀

## 集成方式

### 在训练脚本中自动集成

所有训练脚本现在会在训练完成后自动调用 `create_comprehensive_prototype_analysis()` 函数，生成完整的原型分析报告。

### 手动调用示例

```python
import numpy as np
from utils.visualization import (
    plot_maximum_weight_distribution,
    plot_average_prototype_usage,
    plot_tsne_sample_weights,
    create_comprehensive_prototype_analysis
)

# 加载权重数据
weights = np.load("path/to/sample_weights.npy")

# 单独调用各个函数
plot_maximum_weight_distribution(weights, save_path="max_weight_dist.png")
plot_average_prototype_usage(weights, save_path="prototype_usage.png")
plot_tsne_sample_weights(weights, save_path="tsne_weights.png")

# 或者一次性生成所有分析
create_comprehensive_prototype_analysis(weights, save_dir="./analysis_output")
```

## 输出文件

训练完成后，在输出目录中会生成以下新文件：
- `prototype_analysis_max_weight_distribution.png`: 最大权重分布图
- `prototype_analysis_prototype_usage.png`: 原型使用情况图
- `prototype_analysis_tsne.png`: t-SNE可视化图
- `prototype_analysis_weight_distribution.png`: 权重分布直方图（按原型分组）

## 依赖要求

- matplotlib
- numpy  
- scikit-learn (用于t-SNE功能)
- torch (用于处理PyTorch张量)

如果没有安装scikit-learn，t-SNE功能会自动跳过并提示安装。

## 注意事项

1. t-SNE计算可能较慢，特别是对于大量样本。默认情况下会随机采样5000个样本进行可视化。
2. 所有函数都支持PyTorch张量和NumPy数组输入。
3. 图表会自动关闭以节省内存，只保存到文件中。
