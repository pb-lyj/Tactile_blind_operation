"""
触觉表征学习包

提供三种表征学习方法：
- VQ-VAE: 离散表征学习
- MAE: 掩码自编码器  
- BYOL: 自监督对比学习
"""

__version__ = "1.0.0"
__author__ = "Tactile Representation Team"

# 导入主要组件
from .datasets.tactile_dataset import TactileRepresentationDataset, TactileDataModule
from .models.vqvae_model import create_tactile_vqvae, TactileVQVAE, TactileVQVAEClassifier
from .models.mae_model import create_tactile_mae, TactileMAE, TactileMAEClassifier
from .models.byol_model import create_tactile_byol, TactileBYOL, TactileBYOLClassifier

__all__ = [
    # 数据集
    'TactileRepresentationDataset',
    'TactileDataModule',
    
    # VQ-VAE
    'create_tactile_vqvae',
    'TactileVQVAE', 
    'TactileVQVAEClassifier',
    
    # MAE
    'create_tactile_mae',
    'TactileMAE',
    'TactileMAEClassifier',
    
    # BYOL
    'create_tactile_byol', 
    'TactileBYOL',
    'TactileBYOLClassifier',
]
