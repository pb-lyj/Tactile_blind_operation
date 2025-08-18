# 触觉数据集归一化方法说明

## 概述

在 `TactileForcesDataset` 中融合了多种归一化方法，参考了UniT项目中的图像处理方式，提供了三种不同的归一化策略以适应不同的训练需求。

## 归一化方法

### 1. quantile_zscore (默认方法)
**来源**: 原有方法，适合处理包含极值的触觉力数据
**特点**:
- 使用1%和99%分位数截断极值
- 按通道分别进行Z-score标准化
- 输出数据均值接近0，标准差接近1

**适用场景**:
- 神经网络训练，特别是需要稳定梯度的情况
- 数据包含异常值或极值
- 需要保持数据统计特性的场景

**数学公式**:
```python
# 对每个通道
q1, q99 = percentile(data, [1, 99])
clipped_data = clip(data, q1, q99)
normalized = (clipped_data - mean) / std
```

### 2. minmax_255 (图像处理风格)
**来源**: 参考UniT项目中的图像处理方式
**特点**:
- 先映射到0-255范围（模拟uint8图像）
- 再除以255转换为[0,1]范围的float32
- 类似于标准的图像预处理流程

**适用场景**:
- 与图像数据混合训练
- 需要与预训练的图像模型兼容
- 希望保持数据的非负性

**数学公式**:
```python
# 全局归一化
scaled = (data - data.min()) / (data.max() - data.min()) * 255.0
normalized = scaled.astype(float32) / 255.0
```

### 3. channel_wise (通道独立)
**来源**: 参考深度学习中常见的通道独立归一化
**特点**:
- 每个通道独立进行Min-Max归一化
- 保持通道间的相对关系
- 输出范围严格控制在[0,1]

**适用场景**:
- 不同通道代表不同物理量（如xyz轴力）
- 需要保持通道间独立性
- 适合卷积神经网络

**数学公式**:
```python
# 对每个通道独立
for channel in channels:
    normalized[channel] = (data[channel] - min) / (max - min)
```

## 使用示例

```python
from tactile_representation.Prototype_Discovery.datasets.tactile_dataset import TactileForcesDataset

# 1. 默认方法（分位数+Z-score）
dataset1 = TactileForcesDataset(
    data_root="path/to/data",
    normalize_method='quantile_zscore'  # 默认值
)

# 2. 图像处理风格
dataset2 = TactileForcesDataset(
    data_root="path/to/data",
    normalize_method='minmax_255'
)

# 3. 通道独立归一化
dataset3 = TactileForcesDataset(
    data_root="path/to/data",
    normalize_method='channel_wise'
)
```

## 性能对比

| 方法 | 输出范围 | 统计特性 | 极值处理 | 适用模型 |
|------|----------|----------|----------|----------|
| quantile_zscore | (-∞, +∞) | 均值≈0, std≈1 | 截断处理 | 深度神经网络 |
| minmax_255 | [0, 1] | 保持原始分布形状 | 保留所有值 | 图像模型、预训练模型 |
| channel_wise | [0, 1] | 每通道独立 | 保留所有值 | CNN、多模态模型 |

## 融合的技术细节

### 从UniT项目学习的要点:
1. **数据类型处理**: 参考了uint8→float32的转换流程
2. **范围控制**: 学习了严格的[0,1]范围控制
3. **内存优化**: 借鉴了`astype(np.float32)`的内存管理
4. **安全性检查**: 加入了NaN/Inf检测和后备方案

### 代码健壮性:
- 除零保护: `if std > 1e-8`
- 数值稳定性: `+ 1e-8` 避免除零
- 异常处理: 自动回退到简单归一化
- 类型安全: 明确的数据类型转换

## 建议用法

**训练阶段**:
- 原型自编码器: `quantile_zscore` (默认)
- 图像风格模型: `minmax_255`
- 多通道CNN: `channel_wise`

**推理阶段**:
- 保持与训练时相同的归一化方法
- 确保数据预处理的一致性

**实验对比**:
- 可以通过修改 `normalize_method` 参数快速切换方法
- 建议对比不同方法的训练效果
- 记录最佳归一化方法用于最终模型
