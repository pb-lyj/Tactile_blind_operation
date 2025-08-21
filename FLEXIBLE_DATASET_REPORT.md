# 灵活数据集索引设计报告

## 📋 概述

成功设计并实现了灵活的数据集索引系统，支持时序和无时序两种模式，满足不同训练需求的数据加载方式。

## 🏗️ 设计架构

### 📊 数据结构理解
根据数据验证结果，每个轨迹文件夹包含：
```
action:           (N, 7)     # N个动作（包含时间戳）
end_states:       (N+1, 7)   # N+1个状态（包含时间戳）
forces_l/r:       (N+1, 3, 20, 20)  # N+1个触觉力数据
resultant_*:      (N+1, 3)   # N+1个合力/力矩数据
```

### 🔄 两种工作模式

#### 1. **无时序模式** (`sequence_mode=False`)
- **原理**: 将所有轨迹的对应数据打乱加载
- **索引**: 每个样本是单个时间步的对齐数据
- **适用**: 单步预测、行为克隆、状态-动作对学习

```python
# 无时序模式样本结构
sample = {
    'action': torch.Size([6]),           # 单步动作
    'current_state': torch.Size([6]),    # 当前状态  
    'next_state': torch.Size([6]),       # 下一状态
    'forces_l': torch.Size([3, 20, 20]), # 当前触觉力
    'category': 'cir_lar',
    'trajectory_id': '...',
    'step_idx': 86
}
```

#### 2. **时序模式** (`sequence_mode=True`)
- **原理**: 从单条轨迹中连续读取L步，不跨轨迹
- **索引**: 每个样本是长度为L的序列片段
- **适用**: RNN/LSTM训练、序列预测、时序建模

```python
# 时序模式样本结构 (sequence_length=5)
sample = {
    'actions': torch.Size([5, 6]),        # 5步动作序列
    'states': torch.Size([6, 6]),         # 6个状态（初始状态+5步后状态）
    'forces_l': torch.Size([6, 3, 20, 20]), # 6帧触觉力数据
    'category': 'cir_lar',
    'trajectory_id': '...',
    'start_idx': 0,
    'end_idx': 5,
    'sequence_length': 5
}
```

## 🔧 核心特性

### ✅ 数据对齐保证
1. **严格的一一对应**: 每个时间步的所有数据完全对齐
2. **轨迹完整性验证**: 自动检查数据文件完整性
3. **长度一致性**: 确保不同模态数据在时间维度匹配

### ✅ 灵活索引机制
1. **无时序打乱**: 支持跨轨迹随机采样，提高数据多样性
2. **时序连续性**: 保证序列片段来自同一轨迹，维持时间依赖
3. **长度不足处理**: 自动丢弃长度不足的序列片段

### ✅ 智能数据处理
1. **归一化集成**: 支持不同数据类型的独立归一化
2. **内存效率**: 按需加载，避免预加载大量数据
3. **错误容错**: 自动跳过损坏或缺失的数据文件

## 📊 数据统计结果

### 无时序模式
```
轨迹数量: 40 (训练) + 10 (测试)
样本数量: 11,236 (训练) + 2,543 (测试)
数据类型: action, end_states, forces, resultants
```

### 时序模式 (序列长度=5)
```
轨迹数量: 40 (训练) + 10 (测试)  
样本数量: 11,076 (训练) + 2,503 (测试)
数据类型: action, end_states, forces
```

## 🔗 API接口

### 创建数据集
```python
from policy_learn.dataset_dataloader.flexible_policy_dataset import create_flexible_datasets

# 无时序模式
train_dataset, test_dataset = create_flexible_datasets(
    data_root='/path/to/data',
    categories=['cir_lar', 'rect_med'],
    sequence_mode=False,  # 无时序模式
    normalize_config={
        'actions': 'minmax',
        'forces': 'zscore'
    }
)

# 时序模式
seq_train, seq_test = create_flexible_datasets(
    data_root='/path/to/data',
    sequence_mode=True,    # 时序模式
    sequence_length=10,    # 序列长度
    normalize_config={
        'actions': 'minmax',
        'forces': 'zscore'
    }
)
```

### DataLoader兼容
```python
from torch.utils.data import DataLoader

# 无时序模式
loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# batch['action'].shape = (32, 6)

# 时序模式  
seq_loader = DataLoader(seq_train, batch_size=16, shuffle=True)
# batch['actions'].shape = (16, 10, 6)
```

## 💡 设计优势

### 1. **统一接口**
- 相同的API支持两种不同的数据访问模式
- 无需修改下游代码即可切换模式

### 2. **数据完整性**
- 自动验证数据文件完整性
- 确保不同传感器数据的时间对齐

### 3. **内存高效**
- 按需加载数据，避免内存浪费
- 支持大规模数据集的训练

### 4. **扩展性**
- 易于添加新的数据类型
- 支持自定义归一化方法

## 🎯 应用场景

### 无时序模式适用于:
- ✅ 单步动作预测（如Feature-MLP）
- ✅ 状态-动作映射学习
- ✅ 行为克隆基础训练
- ✅ 数据增强和预处理

### 时序模式适用于:
- ✅ RNN/LSTM序列建模
- ✅ Transformer时序预测
- ✅ 动态系统学习
- ✅ 轨迹规划和优化

## 📈 性能验证

### ✅ 功能测试通过
- 数据加载正确性 ✓
- 索引一致性 ✓  
- DataLoader兼容性 ✓
- 归一化功能 ✓

### ✅ 数据对齐验证
- 单时间步数据对齐 ✓
- 序列数据连续性 ✓
- 跨模态数据同步 ✓

## 🔄 使用示例

```python
# 示例1: 单步预测训练
dataset = FlexiblePolicyDataset(
    data_root='/path/to/data',
    sequence_mode=False,
    normalize_config={'actions': 'minmax', 'forces': 'zscore'}
)

# 示例2: 序列建模训练  
seq_dataset = FlexiblePolicyDataset(
    data_root='/path/to/data',
    sequence_mode=True,
    sequence_length=20,
    normalize_config={'actions': 'minmax', 'forces': 'zscore'}
)

# 示例3: 自定义数据加载
custom_dataset = FlexiblePolicyDataset(
    data_root='/path/to/data',
    use_end_states=False,  # 不加载状态数据
    use_resultants=False,  # 不加载合力数据
    sequence_mode=True,
    sequence_length=15
)
```

## 📋 总结

FlexiblePolicyDataset成功实现了：
1. ✅ **完美数据对齐**: 所有数据类型严格一一对应
2. ✅ **双模式支持**: 时序和无时序模式无缝切换
3. ✅ **智能索引**: 自动处理轨迹边界和长度限制
4. ✅ **高度灵活**: 支持任意数据类型组合和归一化方案
5. ✅ **生产就绪**: 完整的错误处理和性能优化

这个设计为后续的机器人学习算法提供了强大而灵活的数据基础设施！
