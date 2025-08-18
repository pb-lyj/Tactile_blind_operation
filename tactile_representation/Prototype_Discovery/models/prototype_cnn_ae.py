# improved_prototype_ae.py
# 专门为力数据优化的原型自编码器

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        return F.relu(out)
    
    
class ImprovedForcePrototypeAE(nn.Module):
    def __init__(self, num_prototypes=8, input_shape=(3, 20, 20)):
        super().__init__()
        self.K = num_prototypes
        self.C, self.H, self.W = input_shape

        # 改进的原型初始化 - 使用Xavier初始化并添加小的随机偏移
        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, self.C, self.H, self.W))
        nn.init.xavier_normal_(self.prototypes, gain=0.1)  # 小幅度初始化
        
        # 改进的CNN编码器 - 更深的网络和更好的正则化
        # self.encoder = nn.Sequential(
        #     # 第一层
        #     nn.Conv2d(self.C, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Dropout2d(0.1),
            
        #     # 第二层
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Dropout2d(0.1),
            
        #     # 第三层
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
            
        #     # 全局平均池化
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
            
        #     # 全连接层
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(64, num_prototypes),
        #     nn.Softmax(dim=-1)  # 确保权重和为1
        # )
                # 编码器：加入残差块
        self.encoder = nn.Sequential(
            ResidualBlock(self.C, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_prototypes),
            nn.Sigmoid()
        )

        # 改进的权重初始化
        self._init_weights()

    def _init_weights(self):
        """改进的权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码得到原型权重
        weights = self.encoder(x)  # (B, K)
        
        # 总是添加小的噪声防止权重过于集中（训练和推理都需要）
        noise = torch.randn_like(weights) * 0.01
        weights = weights + noise
        weights = F.softmax(weights, dim=-1)  # 重新归一化
        
        B = x.size(0)

        # 扩展原型张量
        protos = self.prototypes.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, K, C, H, W)
        weights_exp = weights.view(B, self.K, 1, 1, 1)

        # 加权重构
        recon = (weights_exp * protos).sum(dim=1)  # (B, C, H, W)
        
        return recon, weights, protos

def compute_improved_losses(x, recon, weights, protos, 
                          diversity_lambda=1.0, entropy_lambda=0.1, 
                          sparsity_lambda=0.01):
    """
    改进的损失函数
    """
    # 1. 重构损失 - 使用Huber损失，对异常值更鲁棒
    recon_loss = F.smooth_l1_loss(recon, x)
    
    # 2. 改进的多样性损失
    B, K = weights.shape
    # 计算原型间的余弦相似度
    protos_flat = protos.mean(dim=0).view(K, -1)  # (K, C*H*W)
    protos_norm = F.normalize(protos_flat, dim=1)
    
    # 相似度矩阵
    sim_matrix = torch.matmul(protos_norm, protos_norm.T)
    # 去除对角线元素
    mask = ~torch.eye(K, dtype=bool, device=sim_matrix.device)
    off_diag_sim = sim_matrix[mask]
    
    # 惩罚高相似度
    diversity_loss = torch.clamp(off_diag_sim, min=0).pow(2).mean()
    
    # 3. 改进的熵损失 - 鼓励适度的权重分布
    # 使用KL散度惩罚过于均匀或过于集中的分布
    uniform_dist = torch.ones_like(weights) / K
    entropy_loss = F.kl_div(torch.log(weights + 1e-8), uniform_dist, reduction='batchmean')
    
    # 4. 改进的稀疏性损失 - 鼓励每个样本主要使用少数几个原型
    # 使用基尼系数衡量权重分布的不均匀程度，值越大越稀疏
    def gini_coefficient(w):
        """计算基尼系数，衡量分布的不均匀程度，范围[0,1]"""
        sorted_w, _ = torch.sort(w, dim=1, descending=False)  # 升序排列
        n = w.size(1)
        index = torch.arange(1, n + 1, dtype=torch.float32, device=w.device)
        return ((2 * index - n - 1) * sorted_w).sum(dim=1) / (n * sorted_w.sum(dim=1) + 1e-8)
    
    # 我们想要高基尼系数（稀疏分布），所以损失是 1 - gini_coefficient
    gini_coeff = gini_coefficient(weights)
    sparsity_loss = (1.0 - gini_coeff).mean()  # 越稀疏损失越小
    
    # 总损失
    total_loss = (recon_loss + 
                 diversity_lambda * diversity_loss + 
                 entropy_lambda * entropy_loss +
                 sparsity_lambda * sparsity_loss)
    
    return total_loss, {
        "recon_loss": recon_loss.item(),
        "diversity_loss": diversity_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "sparsity_loss": sparsity_loss.item(),
        "gini_coeff": gini_coeff.mean().item(),  # 添加基尼系数监控
        "total_loss": total_loss.item()
    }
