"""
MAE (Masked Autoencoder) 模型实现，适配触觉力数据 (3, 20, 20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math


class PatchEmbedding(nn.Module):
    """
    将输入图像转换为patch embeddings
    """
    def __init__(self, img_size=20, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, embed_dim, grid_size, grid_size)
        x = self.projection(x)
        # (B, embed_dim, grid_size, grid_size) -> (B, num_patches, embed_dim)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        # 计算QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 应用注意力权重
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer块"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """MAE编码器"""
    def __init__(self, img_size=20, patch_size=4, in_channels=3, embed_dim=192, 
                 depth=12, num_heads=3, mlp_ratio=4.0, mask_ratio=0.75):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.mask_ratio = mask_ratio
        
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        # 初始化位置编码
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化线性层
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        随机遮罩patches
        Args:
            x: (B, N, D)
            mask_ratio: 遮罩比例
        Returns:
            x_masked: 保留的patches
            mask: 遮罩矩阵
            ids_restore: 恢复原始顺序的索引
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留前len_keep个patches
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # 生成遮罩
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, x):
        # patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # 添加位置编码
        x = x + self.pos_embed[:, 1:, :]
        
        # 随机遮罩
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # 添加cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        return x, mask, ids_restore


class MAEDecoder(nn.Module):
    """MAE解码器"""
    def __init__(self, patch_size=4, in_channels=3, embed_dim=192, decoder_embed_dim=96,
                 decoder_depth=4, decoder_num_heads=3, mlp_ratio=4.0):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # 遮罩token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # 解码器位置编码
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, 25 + 1, decoder_embed_dim))  # 5x5 patches + cls
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio) 
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_channels, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # 嵌入tokens
        x = self.decoder_embed(x)
        
        # 添加遮罩tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # 去掉cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # 恢复顺序
        x = torch.cat([x[:, :1, :], x_], dim=1)  # 添加cls token
        
        # 添加位置编码
        x = x + self.decoder_pos_embed
        
        # Transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.decoder_norm(x)
        
        # 预测
        x = self.decoder_pred(x)
        
        # 移除cls token
        x = x[:, 1:, :]
        
        return x


class TactileMAE(nn.Module):
    """
    触觉MAE模型
    """
    def __init__(self, img_size=20, patch_size=4, in_channels=3, embed_dim=192,
                 depth=12, num_heads=3, decoder_embed_dim=96, decoder_depth=4,
                 decoder_num_heads=3, mlp_ratio=4.0, mask_ratio=0.75):
        super().__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        self.encoder = MAEEncoder(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, mask_ratio=mask_ratio
        )
        
        self.decoder = MAEDecoder(
            patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio
        )

    def patchify(self, imgs):
        """
        将图像转换为patches
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_channels))
        return x

    def unpatchify(self, x):
        """
        将patches转换为图像
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_channels, h * p, h * p))
        return imgs

    def forward(self, imgs):
        # 编码
        latent, mask, ids_restore = self.encoder(imgs)
        
        # 解码
        pred = self.decoder(latent, ids_restore)
        
        return {
            'pred': pred,
            'mask': mask,
            'latent': latent,
            'ids_restore': ids_restore
        }
    
    def encode(self, imgs):
        """仅编码，不进行遮罩"""
        x = self.encoder.patch_embed(imgs)
        x = x + self.encoder.pos_embed[:, 1:, :]
        
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        for block in self.encoder.blocks:
            x = block(x)
        
        x = self.encoder.norm(x)
        return x


class TactileMAELoss(nn.Module):
    """
    MAE损失函数
    """
    def __init__(self, patch_size=4, in_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

    def forward(self, imgs, outputs):
        """
        计算MAE损失
        Args:
            imgs: 原始图像 (B, C, H, W)
            outputs: 模型输出字典
        """
        pred = outputs['pred']
        mask = outputs['mask']
        
        # 将图像转换为patches
        target = self.patchify(imgs)
        
        # 计算损失（仅在遮罩区域）
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # 在patch维度上平均
        
        # 仅计算遮罩区域的损失
        loss = (loss * mask).sum() / mask.sum()
        
        return {
            'total_loss': loss,
            'reconstruction_loss': loss,
            'mask_ratio': mask.mean()
        }

    def patchify(self, imgs):
        """将图像转换为patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_channels, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * self.in_channels))
        return x


class TactileMAEClassifier(nn.Module):
    """
    基于MAE的分类器
    """
    def __init__(self, mae_model, num_classes, freeze_mae=True):
        super().__init__()
        self.mae = mae_model
        if freeze_mae:
            for param in self.mae.parameters():
                param.requires_grad = False
        
        # 分类头
        embed_dim = mae_model.encoder.cls_token.shape[-1]
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 提取MAE特征
        with torch.no_grad() if not self.training else torch.enable_grad():
            encoded = self.mae.encode(x)  # (B, N+1, D)
            cls_token = encoded[:, 0]  # 取cls token
        
        # 分类
        return self.classifier(cls_token)


def create_tactile_mae(config=None):
    """
    创建触觉MAE模型的工厂函数
    """
    if config is None:
        config = {
            'img_size': 20,
            'patch_size': 4,
            'in_channels': 3,
            'embed_dim': 192,
            'depth': 6,  # 减少深度以适应小图像
            'num_heads': 3,
            'decoder_embed_dim': 96,
            'decoder_depth': 4,
            'decoder_num_heads': 3,
            'mlp_ratio': 4.0,
            'mask_ratio': 0.75
        }
    
    model = TactileMAE(**config)
    return model


def create_mae_loss(config=None):
    """
    创建MAE损失函数的工厂函数
    """
    if config is None:
        config = {
            'patch_size': 4,
            'in_channels': 3
        }
    
    return TactileMAELoss(**config)
