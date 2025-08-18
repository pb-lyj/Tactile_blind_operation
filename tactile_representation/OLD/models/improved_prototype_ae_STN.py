# improved_prototype_ae_STN.py
# 基于STN的改进原型自编码器，专门为力数据优化

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==================== Improved Spatial Transformer Module ====================
class ImprovedSpatialTransformer(nn.Module):
    def __init__(self, input_size=(20, 20)):
        super().__init__()
        
        # 计算经过卷积后的特征大小
        pool_h = input_size[0] // 4
        pool_w = input_size[1] // 4
        feature_size = 64 * pool_h * pool_w  # 增加通道数到64
        
        # 改进的定位网络 - 使用BatchNorm和更深的结构
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            nn.Flatten(),
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        
        # 改进的权重初始化
        self._init_weights()
        
        # 初始化最后一层为恒等变换
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

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
            elif isinstance(m, nn.Linear) and m is not self.localization[-1]:
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        theta = self.localization(x)  # (B, 6)
        
        # 添加小的随机扰动防止过拟合
        if self.training:
            noise = torch.randn_like(theta) * 0.01
            theta = theta + noise
        
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_trans = F.grid_sample(x, grid, align_corners=True)
        return x_trans, theta

class ImprovedSharedSpatialTransformer(nn.Module):
    def __init__(self, input_size=(20, 20)):
        super().__init__()
        self.input_size = input_size
        
        # 改进的共享特征提取卷积层
        self.shared_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            
            nn.Flatten()
        )
        
        # 计算经过卷积后的特征大小
        pool_h = input_size[0] // 4
        pool_w = input_size[1] // 4
        feature_size = 64 * pool_h * pool_w
        
        # 改进的共享FC层
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # 每个原型独立的最终变换层
        self.loc_heads = nn.ModuleList()
        
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
        
    def add_localization_head(self):
        """添加独立的定位头"""
        head = nn.Linear(32, 6)
        # 初始化为恒等变换
        head.weight.data.zero_()
        head.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.loc_heads.append(head)
        
    def forward(self, x, proto_idx):
        # 共享特征提取
        features = self.shared_features(x)
        # 共享FC层
        features = self.shared_fc(features)
        # 独立变换层
        theta = self.loc_heads[proto_idx](features)
        
        # 添加小的随机扰动防止过拟合
        if self.training:
            noise = torch.randn_like(theta) * 0.01
            theta = theta + noise
            
        theta = theta.view(-1, 2, 3)
        
        # 空间变换
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_trans = F.grid_sample(x, grid, align_corners=True)
        return x_trans, theta

# ==================== Improved Prototype Autoencoder with STN ====================
class ImprovedPrototypeSTNAE(nn.Module):
    def __init__(self, num_prototypes=8, input_shape=(3, 20, 20), share_stn=True):
        super().__init__()
        self.K = num_prototypes
        self.C, self.H, self.W = input_shape
        self.share_stn = share_stn

        # 改进的原型初始化 - 使用Xavier初始化并添加小的随机偏移
        self.prototypes = nn.Parameter(torch.zeros(num_prototypes, self.C, self.H, self.W))
        nn.init.xavier_normal_(self.prototypes, gain=0.1)  # 小幅度初始化

        # 改进的STN模块
        if share_stn:
            self.shared_stn = ImprovedSharedSpatialTransformer(input_size=(self.H, self.W))
            # 为每个原型添加定位头
            for _ in range(num_prototypes):
                self.shared_stn.add_localization_head()
        else:
            self.stn_modules = nn.ModuleList([
                ImprovedSpatialTransformer(input_size=(self.H, self.W)) 
                for _ in range(num_prototypes)
            ])

        # 改进的CNN编码器 - 更深的网络和更好的正则化
        self.encoder = nn.Sequential(
            # 第一层
            nn.Conv2d(self.C, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            
            # 全连接层
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_prototypes),
            nn.Softmax(dim=-1)  # 确保权重和为1
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
        
        # 添加小的随机扰动防止权重过于集中（训练和推理都需要）
        if self.training:
            noise = torch.randn_like(weights) * 0.01
            weights = weights + noise
            weights = F.softmax(weights, dim=-1)  # 重新归一化
        
        B = x.size(0)
        protos = self.prototypes.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, K, C, H, W)

        transformed_protos = []
        thetas = []

        for k in range(self.K):
            proto_k = protos[:, k]  # (B, C, H, W)
            if self.share_stn:
                trans_k, theta_k = self.shared_stn(proto_k, k)
            else:
                trans_k, theta_k = self.stn_modules[k](proto_k)
            transformed_protos.append(trans_k)
            thetas.append(theta_k)

        transformed_protos = torch.stack(transformed_protos, dim=1)  # (B, K, C, H, W)
        thetas = torch.stack(thetas, dim=1)  # (B, K, 2, 3)

        weights_exp = weights.view(B, self.K, 1, 1, 1)
        recon = (weights_exp * transformed_protos).sum(dim=1)  # (B, C, H, W)

        return recon, weights, transformed_protos, thetas

# ==================== Improved Loss Function ====================
def compute_improved_stn_losses(x, recon, weights, transformed_protos, thetas,
                               diversity_lambda=1.0, entropy_lambda=0.1, 
                               sparsity_lambda=0.01, stn_reg_lambda=0.05):
    """
    基于improved_prototype_ae的改进STN损失函数
    """
    # 1. 重构损失 - 使用Huber损失，对异常值更鲁棒
    recon_loss = F.smooth_l1_loss(recon, x)
    
    # 2. 改进的多样性损失
    B, K, C, H, W = transformed_protos.shape
    # 计算变换后原型间的余弦相似度
    protos_flat = transformed_protos.mean(dim=0).view(K, -1)  # (K, C*H*W)
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
    
    # 5. STN正则化损失 - 改进版本
    # 惩罚过大的变换参数，鼓励小幅度变换
    identity = torch.tensor([[1., 0., 0.], [0., 1., 0.]], 
                           device=thetas.device).view(1, 1, 2, 3)
    identity_expanded = identity.expand_as(thetas)
    
    # 计算变换矩阵与恒等变换的差异
    theta_diff = thetas - identity_expanded
    
    # 分别惩罚平移、旋转和缩放
    # 前4个参数是缩放和旋转矩阵，后2个参数是平移
    rotation_scale_loss = F.mse_loss(theta_diff[:, :, :, :2], 
                                   torch.zeros_like(theta_diff[:, :, :, :2]))
    translation_loss = F.mse_loss(theta_diff[:, :, :, 2], 
                                torch.zeros_like(theta_diff[:, :, :, 2]))
    
    # STN总损失
    stn_loss = rotation_scale_loss + 0.5 * translation_loss
    
    # 6. 额外的STN多样性损失 - 鼓励不同原型学习不同的变换
    if K > 1:
        # 计算不同原型间变换参数的相似性
        theta_flat = thetas.view(B, K, -1)  # (B, K, 6)
        theta_mean = theta_flat.mean(dim=0)  # (K, 6)
        theta_norm = F.normalize(theta_mean, dim=1)
        theta_sim = torch.matmul(theta_norm, theta_norm.T)
        theta_mask = ~torch.eye(K, dtype=bool, device=theta_sim.device)
        theta_diversity_loss = torch.clamp(theta_sim[theta_mask], min=0).pow(2).mean()
    else:
        theta_diversity_loss = torch.tensor(0.0, device=thetas.device)
    
    # 总损失
    total_loss = (recon_loss + 
                 diversity_lambda * diversity_loss + 
                 entropy_lambda * entropy_loss +
                 sparsity_lambda * sparsity_loss +
                 stn_reg_lambda * (stn_loss + 0.1 * theta_diversity_loss))
    
    return total_loss, {
        "recon_loss": recon_loss.item(),
        "diversity_loss": diversity_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "sparsity_loss": sparsity_loss.item(),
        "gini_coeff": gini_coeff.mean().item(),
        "stn_loss": stn_loss.item(),
        "theta_diversity_loss": theta_diversity_loss.item(),
        "total_loss": total_loss.item()
    }

# ==================== 简化的损失函数（兼容原版API） ====================
def compute_losses_compatible(x, recon, weights, transformed_protos, thetas,
                            diversity_lambda=0.1, entropy_lambda=10.0, stn_reg_lambda=0.05):
    """
    与原版API兼容的损失函数
    """
    return compute_improved_stn_losses(
        x, recon, weights, transformed_protos, thetas,
        diversity_lambda=diversity_lambda, 
        entropy_lambda=entropy_lambda/100.0,  # 调整比例
        sparsity_lambda=0.01, 
        stn_reg_lambda=stn_reg_lambda
    )
