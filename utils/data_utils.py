"""
数据处理工具 - 用于样本权重分析和数据后处理
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 设置为非交互模式，不弹出图窗
import matplotlib.pyplot as plt
from utils.visualization import (
    create_comprehensive_prototype_analysis
)


def save_sample_weights_and_analysis(model, dataset, output_dir, batch_size=64):
    """
    保存样本权重并进行分析
    Args:
        model: 训练好的模型
        dataset: 数据集
        output_dir: 输出目录
        batch_size: 批大小
    """
    model.eval()
    
    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    all_weights = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="分析样本权重"):
            inputs = batch['image'].cuda()
            
            if hasattr(model, 'encoder'):
                # 对于baseline和improved模型
                _, weights, _ = model(inputs)
            else:
                # 对于STN模型
                _, weights, _, _ = model(inputs)
            
            all_weights.append(weights.cpu().numpy())
            
            # 如果数据集有标签信息，也收集标签
            if hasattr(dataset, 'get_labels'):
                labels = dataset.get_labels(len(all_labels) * batch_size, 
                                          len(all_labels) * batch_size + batch.size(0))
                all_labels.extend(labels)
    
    # 合并所有权重
    weights_array = np.vstack(all_weights)
    
    # 保存权重
    weights_path = os.path.join(output_dir, "sample_weights.npy")
    np.save(weights_path, weights_array)
    print(f"✅ 样本权重已保存到 {weights_path}")
    
    # 分析权重分布
    analyze_weight_distribution(weights_array, output_dir)
    
    # 绘制权重使用情况（旧版本）
    plot_prototype_usage(weights_array, output_dir)
    
    # 新增：创建全面的原型分析可视化
    create_comprehensive_prototype_analysis(
        weights_array, 
        save_dir=output_dir,
        prefix="prototype_analysis"
    )
    
    return weights_array


def analyze_weight_distribution(weights, output_dir):
    """
    分析权重分布
    Args:
        weights: 权重数组，形状为(N, K)
        output_dir: 输出目录
    """
    N, K = weights.shape
    
    # 计算统计信息
    stats = {
        'mean': np.mean(weights, axis=0),
        'std': np.std(weights, axis=0),
        'min': np.min(weights, axis=0),
        'max': np.max(weights, axis=0),
        'median': np.median(weights, axis=0)
    }
    
    # 计算原型使用频率（权重>0.1的样本比例）
    usage_freq = np.mean(weights > 0.1, axis=0)
    
    # 计算Gini系数（衡量权重不平等程度）
    gini_coeffs = []
    for k in range(K):
        sorted_weights = np.sort(weights[:, k])
        index = np.arange(1, N + 1)
        gini = (2 * np.sum(index * sorted_weights)) / (N * np.sum(sorted_weights)) - (N + 1) / N
        gini_coeffs.append(gini)
    
    # 保存分析结果
    analysis_path = os.path.join(output_dir, "weight_analysis.txt")
    with open(analysis_path, 'w') as f:
        f.write("权重分布分析报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("原型统计信息:\n")
        for k in range(K):
            f.write(f"原型 {k+1}:\n")
            f.write(f"  均值: {stats['mean'][k]:.4f}\n")
            f.write(f"  标准差: {stats['std'][k]:.4f}\n")
            f.write(f"  最小值: {stats['min'][k]:.4f}\n")
            f.write(f"  最大值: {stats['max'][k]:.4f}\n")
            f.write(f"  中位数: {stats['median'][k]:.4f}\n")
            f.write(f"  使用频率 (>0.1): {usage_freq[k]:.2%}\n")
            f.write(f"  Gini系数: {gini_coeffs[k]:.4f}\n")
            f.write("\n")
        
        f.write("整体统计:\n")
        f.write(f"总样本数: {N}\n")
        f.write(f"原型数量: {K}\n")
        f.write(f"平均权重: {np.mean(weights):.4f}\n")
        f.write(f"权重标准差: {np.std(weights):.4f}\n")
        f.write(f"平均Gini系数: {np.mean(gini_coeffs):.4f}\n")
    
    print(f"✅ 权重分析报告已保存到 {analysis_path}")


def plot_prototype_usage(weights, output_dir):
    """
    绘制原型使用情况图表
    Args:
        weights: 权重数组，形状为(N, K)
        output_dir: 输出目录
    """
    N, K = weights.shape
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 权重分布直方图
    ax1 = axes[0, 0]
    for k in range(K):
        ax1.hist(weights[:, k], bins=50, alpha=0.6, label=f'Prototype {k+1}', density=True)
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Weight Distribution by Prototype')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 原型使用频率条形图
    ax2 = axes[0, 1]
    usage_freq = np.mean(weights > 0.1, axis=0)
    bars = ax2.bar(range(1, K+1), usage_freq)
    ax2.set_xlabel('Prototype')
    ax2.set_ylabel('Usage Frequency (>0.1)')
    ax2.set_title('Prototype Usage Frequency')
    ax2.set_xticks(range(1, K+1))
    ax2.grid(True, alpha=0.3)
    
    # 在条形图上添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}', ha='center', va='bottom')
    
    # 3. 平均权重热力图
    ax3 = axes[1, 0]
    mean_weights = np.mean(weights, axis=0).reshape(1, -1)
    im = ax3.imshow(mean_weights, cmap='viridis', aspect='auto')
    ax3.set_xlabel('Prototype')
    ax3.set_ylabel('Mean Weight')
    ax3.set_title('Average Weight per Prototype')
    ax3.set_xticks(range(K))
    ax3.set_xticklabels([f'P{i+1}' for i in range(K)])
    ax3.set_yticks([0])
    ax3.set_yticklabels(['Mean'])
    plt.colorbar(im, ax=ax3)
    
    # 4. 权重相关性矩阵
    ax4 = axes[1, 1]
    corr_matrix = np.corrcoef(weights.T)
    im = ax4.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax4.set_xlabel('Prototype')
    ax4.set_ylabel('Prototype')
    ax4.set_title('Prototype Weight Correlation')
    ax4.set_xticks(range(K))
    ax4.set_xticklabels([f'P{i+1}' for i in range(K)])
    ax4.set_yticks(range(K))
    ax4.set_yticklabels([f'P{i+1}' for i in range(K)])
    plt.colorbar(im, ax=ax4)
    
    # 在相关性矩阵中添加数值
    for i in range(K):
        for j in range(K):
            text = ax4.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")
    
    plt.tight_layout()
    
    # 保存图表
    usage_path = os.path.join(output_dir, "prototype_usage.png")
    plt.savefig(usage_path, dpi=150, bbox_inches='tight')
    print(f"✅ 原型使用分析图已保存到 {usage_path}")
    
    plt.close()  # 关闭图形以释放内存


def calculate_prototype_diversity(prototypes):
    """
    计算原型多样性指标
    Args:
        prototypes: 原型张量，形状为(K, C, H, W)
    Returns:
        dict: 多样性指标字典
    """
    if isinstance(prototypes, torch.Tensor):
        prototypes = prototypes.detach().cpu().numpy()
    
    K, C, H, W = prototypes.shape
    
    # 展平原型
    protos_flat = prototypes.reshape(K, -1)
    
    # 计算余弦相似度矩阵
    from sklearn.metrics.pairwise import cosine_similarity
    cos_sim_matrix = cosine_similarity(protos_flat)
    
    # 去除对角线（自相似度=1）
    mask = np.eye(K, dtype=bool)
    off_diag_similarities = cos_sim_matrix[~mask]
    
    # 计算多样性指标
    diversity_metrics = {
        'mean_similarity': np.mean(off_diag_similarities),
        'std_similarity': np.std(off_diag_similarities),
        'min_similarity': np.min(off_diag_similarities),
        'max_similarity': np.max(off_diag_similarities),
        'diversity_score': 1 - np.mean(np.abs(off_diag_similarities))  # 1 - 平均绝对相似度
    }
    
    return diversity_metrics
