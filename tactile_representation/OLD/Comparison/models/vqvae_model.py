"""
VQ-VAE 模型实现，适配触觉力数据 (3, 20, 20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):
    """
    向量量化模块
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, inputs):
        # inputs: (B, C, H, W)
        input_shape = inputs.shape
        
        # 展平输入
        flat_input = inputs.view(-1, self.embedding_dim)
        
        # 计算距离
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # 编码
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # 损失
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 直通估计器
        quantized = inputs + (quantized - inputs).detach()
        
        return quantized, loss, encoding_indices.view(input_shape[0], -1)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        return F.relu(out + skip)


class TactileEncoder(nn.Module):
    """触觉数据编码器"""
    def __init__(self, in_channels=3, hidden_dim=128, latent_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # (3, 20, 20) -> (64, 20, 20)
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            
            # (64, 20, 20) -> (64, 10, 10)
            ResidualBlock(64, 64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            
            # (64, 10, 10) -> (128, 10, 10)
            ResidualBlock(64, 128),
            
            # (128, 10, 10) -> (128, 5, 5)
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.ReLU(),
            
            # (128, 5, 5) -> (latent_dim, 5, 5)
            ResidualBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, latent_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)


class TactileDecoder(nn.Module):
    """触觉数据解码器"""
    def __init__(self, latent_dim=64, hidden_dim=128, out_channels=3):
        super().__init__()
        
        self.decoder = nn.Sequential(
            # (latent_dim, 5, 5) -> (128, 5, 5)
            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            ResidualBlock(hidden_dim, hidden_dim),
            
            # (128, 5, 5) -> (128, 10, 10)
            nn.ConvTranspose2d(hidden_dim, 128, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            ResidualBlock(128, 128),
            
            # (128, 10, 10) -> (64, 20, 20)
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            
            # (64, 20, 20) -> (3, 20, 20)
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


class TactileVQVAE(nn.Module):
    """
    触觉VQ-VAE模型
    """
    def __init__(self, in_channels=3, latent_dim=64, num_embeddings=512, 
                 commitment_cost=0.25, hidden_dim=128):
        super().__init__()
        
        self.encoder = TactileEncoder(in_channels, hidden_dim, latent_dim)
        self.quantizer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = TactileDecoder(latent_dim, hidden_dim, in_channels)

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        
        # 量化
        quantized, vq_loss, encoding_indices = self.quantizer(encoded)
        
        # 解码
        reconstructed = self.decoder(quantized)
        
        return {
            'reconstructed': reconstructed,
            'vq_loss': vq_loss,
            'encoded': encoded,
            'quantized': quantized,
            'encoding_indices': encoding_indices
        }
    
    def encode(self, x):
        """编码到潜在空间"""
        return self.encoder(x)
    
    def quantize(self, x):
        """量化潜在表示"""
        return self.quantizer(x)
    
    def decode(self, x):
        """从潜在空间解码"""
        return self.decoder(x)
    
    def encode_to_indices(self, x):
        """编码为离散索引"""
        encoded = self.encoder(x)
        _, _, indices = self.quantizer(encoded)
        return indices
    
    def decode_from_indices(self, indices, shape):
        """从离散索引解码"""
        # 重建量化向量
        flat_indices = indices.view(-1)
        quantized_flat = self.quantizer.embedding(flat_indices)
        quantized = quantized_flat.view(shape)
        
        # 解码
        return self.decoder(quantized)


class TactileVQVAELoss(nn.Module):
    """
    VQ-VAE损失函数
    """
    def __init__(self, reconstruction_weight=1.0, vq_weight=1.0, perceptual_weight=0.0):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.vq_weight = vq_weight
        self.perceptual_weight = perceptual_weight

    def forward(self, inputs, outputs):
        """
        Args:
            inputs: 原始输入 (B, C, H, W)
            outputs: 模型输出字典
        """
        reconstructed = outputs['reconstructed']
        vq_loss = outputs['vq_loss']
        
        # 重建损失
        recon_loss = F.mse_loss(reconstructed, inputs)
        
        # 总损失
        total_loss = (self.reconstruction_weight * recon_loss + 
                     self.vq_weight * vq_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'vq_loss': vq_loss,
            'perceptual_loss': torch.tensor(0.0).to(total_loss.device)  # 占位符
        }


# 用于分类任务的特征提取器
class TactileVQVAEClassifier(nn.Module):
    """
    基于VQ-VAE的分类器
    """
    def __init__(self, vqvae_model, num_classes, freeze_vqvae=True):
        super().__init__()
        self.vqvae = vqvae_model
        if freeze_vqvae:
            for param in self.vqvae.parameters():
                param.requires_grad = False
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 256),  # 假设latent_dim=64
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 提取VQ-VAE特征
        with torch.no_grad() if not self.training else torch.enable_grad():
            encoded = self.vqvae.encode(x)
        
        # 分类
        return self.classifier(encoded)


def create_tactile_vqvae(config=None):
    """
    创建触觉VQ-VAE模型的工厂函数
    """
    if config is None:
        config = {
            'in_channels': 3,
            'latent_dim': 64,
            'num_embeddings': 512,
            'commitment_cost': 0.25,
            'hidden_dim': 128
        }
    
    model = TactileVQVAE(**config)
    return model


def create_vqvae_loss(config=None):
    """
    创建VQ-VAE损失函数的工厂函数
    """
    if config is None:
        config = {
            'reconstruction_weight': 1.0,
            'vq_weight': 1.0,
            'perceptual_weight': 0.0
        }
    
    return TactileVQVAELoss(**config)
