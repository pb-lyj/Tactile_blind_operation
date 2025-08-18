# 触觉表征学习项目

这个项目实现了三种不同的触觉表征学习方法，用于处理触觉力数据 (3, 20, 20)：

1. **VQ-VAE** (Vector Quantized Variational Autoencoder) - 离散表征学习
2. **MAE** (Masked Autoencoder) - 掩码自编码器
3. **BYOL** (Bootstrap Your Own Latent) - 自监督对比学习

## 项目结构

```
tactile_representation/
├── datasets/
│   └── tactile_dataset.py          # 数据集加载和处理
├── models/
│   ├── vqvae_model.py              # VQ-VAE模型实现
│   ├── mae_model.py                # MAE模型实现
│   └── byol_model.py               # BYOL模型实现
├── training/
│   ├── train_vqvae.py              # VQ-VAE训练脚本
│   ├── train_mae.py                # MAE训练脚本
│   └── train_byol.py               # BYOL训练脚本
├── main_train.py                   # 统一训练入口
└── README.md                       # 项目说明
```

## 数据格式

项目使用的触觉数据格式：
- 输入：(3, 20, 20) 张量，表示触觉传感器的3个通道和20x20像素
- 数据源：左右手触觉传感器的力数据

## 快速开始

### 1. 环境要求

```bash
pip install torch torchvision numpy matplotlib tqdm scikit-learn einops
```

### 2. 数据准备

确保数据按以下结构组织：
```
data/data25.7_aligned/
├── cir_lar/
├── cir_med/
├── cir_sma/
├── rect_lar/
├── rect_med/
├── rect_sma/
├── tri_lar/
├── tri_med/
└── tri_sma/
```

每个类别文件夹包含时间戳命名的子文件夹，每个子文件夹包含：
- `_forces_l.npy`: 左手力数据
- `_forces_r.npy`: 右手力数据

### 3. 训练模型

#### 使用统一训练脚本（推荐）

```bash
# 训练VQ-VAE
python main_train.py vqvae --data_root ./data/data25.7_aligned --epochs 100

# 训练MAE
python main_train.py mae --data_root ./data/data25.7_aligned --epochs 200

# 训练BYOL
python main_train.py byol --data_root ./data/data25.7_aligned --epochs 300
```

#### 生成配置文件模板

```bash
# 生成VQ-VAE配置模板
python main_train.py vqvae --generate_config

# 生成MAE配置模板  
python main_train.py mae --generate_config

# 生成BYOL配置模板
python main_train.py byol --generate_config
```

#### 使用配置文件训练

```bash
# 编辑配置文件后训练
python main_train.py vqvae --config config_vqvae_template.json
```

#### 指定特定类别训练

```bash
# 只训练圆形大号和矩形中号
python main_train.py vqvae --categories cir_lar rect_med
```

### 4. 单独使用各个训练脚本

```bash
# VQ-VAE训练
cd training
python train_vqvae.py --data_root ../data/data25.7_aligned --epochs 100

# MAE训练
python train_mae.py --data_root ../data/data25.7_aligned --epochs 200 --mask_ratio 0.75

# BYOL训练
python train_byol.py --data_root ../data/data25.7_aligned --epochs 300
```

## 模型特点

### VQ-VAE
- **优势**: 学习离散表征，适合生成任务
- **应用**: 触觉数据压缩、生成、风格转换
- **特点**: 量化瓶颈、重建质量高

### MAE
- **优势**: 强大的自监督学习能力
- **应用**: 特征提取、预训练、下游任务微调
- **特点**: 掩码重建、Transformer架构

### BYOL
- **优势**: 不需要负样本的对比学习
- **应用**: 表征学习、相似度计算、聚类
- **特点**: 指数移动平均、数据增强

## 输出结果

训练完成后，每个方法会在 `./outputs/` 目录下生成：

```
outputs/
├── vqvae_YYYYMMDD_HHMMSS/
│   ├── config.json                 # 训练配置
│   ├── best_model.pth              # 最佳模型
│   ├── final_model.pth             # 最终模型
│   ├── training_curves.png         # 训练曲线
│   └── reconstruction_samples.png  # 重建样本可视化
├── mae_YYYYMMDD_HHMMSS/
│   └── ...
└── byol_YYYYMMDD_HHMMSS/
    └── ...
```

## 模型使用

### 加载训练好的模型

```python
import torch
from models.vqvae_model import create_tactile_vqvae

# 加载VQ-VAE模型
model = create_tactile_vqvae()
checkpoint = torch.load('outputs/vqvae_xxx/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理
with torch.no_grad():
    output = model(input_tensor)
    reconstructed = output['reconstructed']
    latent = output['encoded']
```

### 提取特征用于下游任务

```python
# 使用预训练的编码器提取特征
encoder = model.encoder  # VQ-VAE编码器
features = encoder(input_data)

# 或使用MAE编码器
mae_features = mae_model.encode(input_data)
```

## 超参数调优

### 重要超参数说明

**VQ-VAE:**
- `latent_dim`: 潜在空间维度 (推荐: 64-128)
- `num_embeddings`: 词典大小 (推荐: 256-1024)
- `commitment_cost`: 承诺损失权重 (推荐: 0.25)

**MAE:**
- `mask_ratio`: 掩码比例 (推荐: 0.6-0.8)
- `patch_size`: patch大小 (推荐: 4 for 20x20图像)
- `embed_dim`: 嵌入维度 (推荐: 192-384)

**BYOL:**
- `moving_average_decay`: EMA衰减率 (推荐: 0.99-0.999)
- `projection_dim`: 投影维度 (推荐: 128-256)
- 数据增强强度需要调整

## 常见问题

### 1. 显存不足
- 减小batch_size
- 使用梯度累积
- 减小模型维度

### 2. 训练不收敛
- 调整学习率
- 增加warmup
- 检查数据归一化

### 3. 重建质量差
- 增加模型容量
- 调整损失权重
- 增加训练轮数

## 扩展使用

### 分类任务微调

```python
from models.vqvae_model import TactileVQVAEClassifier

# 基于预训练VQ-VAE的分类器
classifier = TactileVQVAEClassifier(
    vqvae_model=pretrained_vqvae,
    num_classes=9,  # 9个形状类别
    freeze_vqvae=True
)
```

### 自定义数据增强

```python
from models.byol_model import TactileDataAugmentation

# 自定义增强策略
augmentation = TactileDataAugmentation(
    noise_std=0.05,
    rotation_angle=10,
    flip_prob=0.3
)
```

## 引用

如果使用此项目，请引用相关论文：
- VQ-VAE: "Neural Discrete Representation Learning"
- MAE: "Masked Autoencoders Are Scalable Vision Learners"  
- BYOL: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
