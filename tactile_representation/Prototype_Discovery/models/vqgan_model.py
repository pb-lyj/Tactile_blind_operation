"""
VQGAN模型 - 基于CNN编码解码器骨干网络的VQ-GAN实现
适配触觉力数据 (3, 20, 20)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cnn_autoencoder import TactileCNNEncoder, TactileCNNDecoder


class VectorQuantizer(nn.Module):
    """
    向量量化模块 - EMA更新版本
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        # 嵌入向量
        embed = torch.randn(embedding_dim, num_embeddings)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, inputs):
        """
        前向传播
        Args:
            inputs: (B, C, H, W) 或 (B, C)
        """
        input_shape = inputs.shape
        
        # 将4D张量展平为2D：(B, C, H, W) -> (B*H*W, C)
        if len(input_shape) == 4:
            B, C, H, W = input_shape
            flat_input = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)
        else:
            # 对于2D输入 (B, C)
            flat_input = inputs
        
        # 计算距离
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                    + torch.sum(self.embed ** 2, dim=0)
                    - 2 * torch.matmul(flat_input, self.embed))
        
        # 编码
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, 
                               device=inputs.device, dtype=inputs.dtype)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 量化
        quantized = torch.matmul(encodings, self.embed.t())
        
        # EMA更新嵌入向量（仅在训练时）
        if self.training:
            self.cluster_size = self.decay * self.cluster_size + (1 - self.decay) * torch.sum(encodings, 0)
            n = torch.sum(self.cluster_size)
            self.cluster_size = (self.cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.embed_avg = self.decay * self.embed_avg + (1 - self.decay) * dw.t()
            self.embed = self.embed_avg / self.cluster_size.unsqueeze(0)
        
        # 计算VQ损失
        e_latent_loss = F.mse_loss(quantized.detach(), flat_input)
        loss = self.commitment_cost * e_latent_loss
        
        # 直通估计器
        quantized = flat_input + (quantized - flat_input).detach()
        
        # 恢复原始形状
        if len(input_shape) == 4:
            quantized = quantized.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # 计算困惑度（衡量码本使用率）
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, encoding_indices.view(-1), perplexity


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN判别器 - 用于对抗训练
    """
    def __init__(self, in_channels=3, ndf=64):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            # (3, 20, 20) -> (64, 10, 10)
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (64, 10, 10) -> (128, 5, 5)
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (128, 5, 5) -> (256, 3, 3)
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (256, 3, 3) -> (1, 2, 2)
            nn.Conv2d(ndf * 4, 1, 3, stride=1, padding=1),
        )
        
    def forward(self, x):
        return self.discriminator(x)


class TactileVQGAN(nn.Module):
    """
    触觉VQGAN模型 - 基于CNN编码解码器
    """
    def __init__(self, config):
        super().__init__()
        
        # 编码器和解码器（使用CNN骨干网络）
        self.encoder = TactileCNNEncoder(
            in_channels=config.get('in_channels', 3),
            latent_dim=config.get('latent_dim', 256)
        )
        
        # 量化前的投影层
        self.pre_quant_conv = nn.Conv2d(256, config.get('embedding_dim', 256), 1)
        
        # 向量量化器
        self.quantizer = VectorQuantizer(
            num_embeddings=config.get('num_embeddings', 1024),
            embedding_dim=config.get('embedding_dim', 256),
            commitment_cost=config.get('commitment_cost', 0.25)
        )
        
        # 量化后的投影层
        self.post_quant_conv = nn.Conv2d(config.get('embedding_dim', 256), 256, 1)
        
        self.decoder = TactileCNNDecoder(
            latent_dim=256 * 5 * 5,  # 匹配编码器输出
            out_channels=config.get('out_channels', 3)
        )
        
    def encode(self, x):
        """编码到量化前的特征"""
        h = self.encoder.encoder(x)  # 使用encoder的卷积部分
        h = self.pre_quant_conv(h)
        return h
    
    def decode(self, quant):
        """从量化特征解码"""
        quant = self.post_quant_conv(quant)
        # 展平并通过解码器
        quant_flat = quant.view(quant.size(0), -1)
        dec = self.decoder(quant_flat)
        return dec
    
    def forward(self, x):
        """完整前向传播"""
        # 编码
        h = self.encode(x)
        
        # 量化
        quant, vq_loss, encoding_indices, perplexity = self.quantizer(h)
        
        # 解码
        reconstructed = self.decode(quant)
        
        return {
            'reconstructed': reconstructed,
            'vq_loss': vq_loss,
            'encoding_indices': encoding_indices,
            'perplexity': perplexity,
            'pre_quant': h,
            'quantized': quant
        }


def compute_vqgan_losses(inputs, outputs, discriminator, config, global_step):
    """
    计算VQGAN损失
    
    Args:
        inputs: 输入数据 (B, 3, 20, 20)
        outputs: 生成器输出
        discriminator: 判别器模型
        config: 损失配置
        global_step: 全局步数
    
    Returns:
        gen_loss: 生成器损失
        disc_loss: 判别器损失
        metrics: 损失分解字典
    """
    reconstructed = outputs['reconstructed']
    vq_loss = outputs['vq_loss']
    perplexity = outputs['perplexity']
    
    # 重建损失
    recon_loss = F.mse_loss(reconstructed, inputs)
    
    # 感知损失（简化版，使用L1损失代替）
    perceptual_loss = F.l1_loss(reconstructed, inputs)
    
    # 判别器输出
    disc_real = discriminator(inputs)
    disc_fake = discriminator(reconstructed)
    
    # 生成器损失（欺骗判别器）
    gen_adv_loss = -torch.mean(disc_fake)
    
    # 判别器损失
    disc_loss = torch.mean(F.relu(1.0 - disc_real)) + torch.mean(F.relu(1.0 + disc_fake))
    
    # 自适应权重
    if global_step < config.get('discriminator_start', 30000):
        disc_weight = 0.0
    else:
        disc_weight = config.get('disc_weight', 0.1)
    
    # 总生成器损失
    gen_loss = (recon_loss + 
                config.get('perceptual_weight', 0.1) * perceptual_loss +
                config.get('vq_weight', 1.0) * vq_loss +
                disc_weight * gen_adv_loss)
    
    metrics = {
        'recon_loss': recon_loss.item(),
        'perceptual_loss': perceptual_loss.item(),
        'vq_loss': vq_loss.item(),
        'gen_adv_loss': gen_adv_loss.item(),
        'disc_loss': disc_loss.item(),
        'perplexity': perplexity.item(),
        'gen_total_loss': gen_loss.item()
    }
    
    return gen_loss, disc_loss, metrics


# 便利函数
def create_tactile_vqgan(config):
    """创建触觉VQGAN模型"""
    generator = TactileVQGAN(config)
    discriminator = PatchGANDiscriminator(
        in_channels=config.get('in_channels', 3),
        ndf=config.get('disc_ndf', 64)
    )
    return generator, discriminator


if __name__ == '__main__':
    # 测试模型
    config = {
        'in_channels': 3,
        'latent_dim': 256,
        'embedding_dim': 256,
        'num_embeddings': 1024,
        'commitment_cost': 0.25
    }
    
    generator, discriminator = create_tactile_vqgan(config)
    
    # 测试输入
    x = torch.randn(4, 3, 20, 20)
    
    # 前向传播
    outputs = generator(x)
    disc_out = discriminator(x)
    
    print(f"输入形状: {x.shape}")
    print(f"重建形状: {outputs['reconstructed'].shape}")
    print(f"判别器输出形状: {disc_out.shape}")
    print(f"困惑度: {outputs['perplexity'].item():.2f}")
    print(f"生成器参数数量: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"判别器参数数量: {sum(p.numel() for p in discriminator.parameters()):,}")
