"""
可视化工具 - 用于绘制损失曲线和保存原型图像
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置为非交互模式，不弹出图窗
import matplotlib.pyplot as plt
import torch


def save_physicalXYZ_images(images, save_path, prefix="tactile"):
    """
    保存物理意义的XYZ触觉力图像
    images: shape (batch_size, 3, H, W)，3个通道分别表示X, Y, Z方向的力
    """
    # 处理输入格式：如果是tensor则转换为numpy，如果已经是numpy则保持不变
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    
    batch_size = images.shape[0]
    
    for i in range(batch_size):
        # 创建2行布局：第一行1个大图，第二行3个小图
        fig = plt.figure(figsize=(15, 10))
        
        # 提取每个通道的数据
        x_force = images[i, 0]  # X方向力
        y_force = images[i, 1]  # Y方向力
        z_force = images[i, 2]  # Z方向力
        
        # 第一行：三通道合并可视化（大图） - 放在正中间
        ax_combined = plt.subplot2grid((2, 3), (0, 1), fig=fig)
        
        # XY方向用箭头表示，Z用背景红色深浅表示
        ax_combined.imshow(z_force, cmap='Reds', alpha=0.7, interpolation='nearest')
        
        # 创建箭头网格
        H, W = x_force.shape
        step = max(1, min(H, W) // 10)  # 箭头间隔，避免过于密集
        y_indices, x_indices = np.meshgrid(
            np.arange(0, H, step),
            np.arange(0, W, step),
            indexing='ij'
        )
        
        # 下采样力数据用于箭头显示
        x_arrows = x_force[::step, ::step]
        y_arrows = y_force[::step, ::step]
        
        # 绘制箭头
        ax_combined.quiver(x_indices, y_indices, x_arrows, y_arrows, 
                          color='blue', alpha=0.8, scale_units='xy', scale=1,
                          width=0.003, headwidth=3, headlength=5)
        
        ax_combined.set_title('Combined XYZ Force Visualization (XY=arrows, Z=background)', fontsize=14, fontweight='bold')
        ax_combined.set_xlabel('Width')
        ax_combined.set_ylabel('Height')
        
        # 第二行：三个单独通道，设置统一的颜色范围
        # 计算所有力数据的最大绝对值，用于统一颜色范围
        max_abs_value = max(np.max(np.abs(x_force)), np.max(np.abs(y_force)), np.max(np.abs(z_force)))
        vmin, vmax = -max_abs_value, max_abs_value
        
        # X方向力 - 使用 RdBu_r 颜色映射（正值红色，负值蓝色）
        ax_x = plt.subplot2grid((2, 3), (1, 0), fig=fig)
        im1 = ax_x.imshow(x_force, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax_x.set_title('X Direction Force')
        ax_x.set_xlabel('Width')
        ax_x.set_ylabel('Height')
        plt.colorbar(im1, ax=ax_x, fraction=0.046, pad=0.04)
        
        # Y方向力 - 使用 RdBu_r 颜色映射（正值红色，负值蓝色）
        ax_y = plt.subplot2grid((2, 3), (1, 1), fig=fig)
        im2 = ax_y.imshow(y_force, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax_y.set_title('Y Direction Force')
        ax_y.set_xlabel('Width')
        ax_y.set_ylabel('Height')
        plt.colorbar(im2, ax=ax_y, fraction=0.046, pad=0.04)
        
        # Z方向力 - 使用 RdBu_r 颜色映射（正值红色，负值蓝色）
        ax_z = plt.subplot2grid((2, 3), (1, 2), fig=fig)
        im3 = ax_z.imshow(z_force, cmap='RdBu_r', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax_z.set_title('Z Direction Force')
        ax_z.set_xlabel('Width')
        ax_z.set_ylabel('Height')
        plt.colorbar(im3, ax=ax_z, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        # 保存图像
        save_file = os.path.join(save_path, f"{prefix}_physical_xyz_{i}.png")
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        plt.close()


def plot_all_losses_single_plot(loss_history, save_path=None, title="Training Loss Curves"):
    """
    在一张图上绘制所有损失曲线
    Args:
        loss_history: 损失历史字典
        save_path: 保存路径
        title: 图表标题
    """
    epochs = loss_history['epoch']
    
    # 获取所有损失键（除了epoch）
    loss_keys = [k for k in loss_history.keys() if k != 'epoch']
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 定义颜色
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, key in enumerate(loss_keys):
        if key in loss_history:
            # 确保数据长度匹配
            if len(epochs) == len(loss_history[key]):
                color = colors[i % len(colors)]
                linewidth = 4 if key == 'total_loss' else 2
                plt.plot(epochs, loss_history[key], 
                        color=color, linewidth=linewidth, label=key.replace('_', ' ').title())
            else:
                print(f"⚠️  警告: {key} 数据长度不匹配 (epochs: {len(epochs)}, {key}: {len(loss_history[key])}), 跳过绘图")
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Value', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    
    # 设置坐标轴刻度
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 添加更多的网格线
    plt.minorticks_on()
    plt.grid(which='minor', alpha=0.2)
    
    # 调整布局以适应图例
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 合并损失曲线已保存到 {save_path}")
    
    plt.close()  # 关闭图形以释放内存


def plot_weight_distribution(weights, save_path=None, title="Weight Distribution Analysis"):
    """
    绘制权重分布分析图 - 包含所有原型的权重分布和最大权重分布
    Args:
        weights: 权重数组，形状为(N, K)，其中N是样本数，K是原型数
        save_path: 保存路径
        title: 图表标题
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    K = weights.shape[1]
    
    # 创建左右两个子图的布局
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 左侧子图：各个原型的权重分布
    if K <= 6:  # 如果原型数量较少，显示所有原型
        colors = plt.cm.Set1(np.linspace(0, 1, K))
        for k in range(K):
            ax_left.hist(weights[:, k], bins=50, alpha=0.6, density=True, 
                        color=colors[k], label=f'Prototype {k+1}', edgecolor='black', linewidth=0.5)
        ax_left.legend(fontsize=12)
    else:  # 如果原型数量太多，只显示前6个
        colors = plt.cm.Set1(np.linspace(0, 1, 6))
        for k in range(6):
            ax_left.hist(weights[:, k], bins=50, alpha=0.6, density=True, 
                        color=colors[k], label=f'Prototype {k+1}', edgecolor='black', linewidth=0.5)
        ax_left.legend(fontsize=12)
        ax_left.text(0.02, 0.98, f'Showing first 6 of {K} prototypes', 
                    transform=ax_left.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax_left.set_title('Individual Prototype Weight Distributions', fontsize=14, fontweight='bold')
    ax_left.set_xlabel('Weight Value', fontsize=12, fontweight='bold')
    ax_left.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax_left.grid(True, alpha=0.3)
    ax_left.tick_params(axis='both', which='major', labelsize=11)
    
    # 右侧子图：最大权重分布
    max_weights = np.max(weights, axis=1)
    ax_right.hist(max_weights, bins=50, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # 添加统计信息
    mean_max = np.mean(max_weights)
    std_max = np.std(max_weights)
    ax_right.axvline(mean_max, color='red', linestyle='--', linewidth=2, alpha=0.8, 
                    label=f'Mean: {mean_max:.3f}')
    ax_right.axvline(mean_max + std_max, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                    label=f'Mean + Std: {mean_max + std_max:.3f}')
    ax_right.axvline(mean_max - std_max, color='orange', linestyle=':', linewidth=2, alpha=0.8, 
                    label=f'Mean - Std: {mean_max - std_max:.3f}')
    
    ax_right.set_title('Maximum Weight Distribution', fontsize=14, fontweight='bold')
    ax_right.set_xlabel('Max Activation Value', fontsize=12, fontweight='bold')
    ax_right.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax_right.grid(True, alpha=0.3)
    ax_right.legend(fontsize=10)
    ax_right.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 权重分布分析图已保存到 {save_path}")
    
    plt.close()  # 关闭图形以释放内存


def create_comparison_plot(results_dict, metric='total_loss', save_path=None):
    """
    创建多模型比较图
    Args:
        results_dict: 结果字典，格式为 {model_name: loss_history}
        metric: 要比较的指标
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, loss_history in results_dict.items():
        if loss_history is not None and metric in loss_history:
            epochs = loss_history['epoch']
            values = loss_history[metric]
            plt.plot(epochs, values, label=model_name, linewidth=2)
    
    plt.title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 比较图已保存到 {save_path}")
    
    plt.close()  # 关闭图形以释放内存



    """
    创建多模型比较图
    Args:
        results_dict: 结果字典，格式为 {model_name: loss_history}
        metric: 要比较的指标
        save_path: 保存路径
    """
    plt.figure(figsize=(12, 8))
    
    for model_name, loss_history in results_dict.items():
        if loss_history is not None and metric in loss_history:
            epochs = loss_history['epoch']
            values = loss_history[metric]
            plt.plot(epochs, values, label=model_name, linewidth=2)
    
    plt.title(f'Model Comparison: {metric.replace("_", " ").title()}', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 比较图已保存到 {save_path}")
    
    plt.close()  # 关闭图形以释放内存


def plot_average_prototype_usage(weights, save_path=None, title="Average Prototype Usage"):
    """
    绘制平均原型使用情况柱状图
    Args:
        weights: 权重数组，形状为(N, K)，其中N是样本数，K是原型数
        save_path: 保存路径
        title: 图表标题
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # 计算每个原型的平均使用率
    avg_usage = np.mean(weights, axis=0)
    prototype_indices = np.arange(len(avg_usage))
    
    plt.figure(figsize=(12, 8))
    
    # 绘制柱状图
    bars = plt.bar(prototype_indices, avg_usage, color='steelblue', alpha=0.7, edgecolor='black')
    
    # 在每个柱子上显示数值
    for i, (bar, value) in enumerate(zip(bars, avg_usage)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Prototype Index', fontsize=14, fontweight='bold')
    plt.ylabel('Average Activation', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(prototype_indices, fontsize=12)
    plt.yticks(fontsize=12)
    
    # 设置y轴范围，留出显示数值的空间
    plt.ylim(0, max(avg_usage) * 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ 平均原型使用图已保存到 {save_path}")
    
    plt.close()  # 关闭图形以释放内存


def plot_tsne_sample_weights(weights, save_path=None, title="t-SNE of Sample Weights", n_samples=5000):
    """
    绘制样本权重的t-SNE可视化
    Args:
        weights: 权重数组，形状为(N, K)，其中N是样本数，K是原型数
        save_path: 保存路径
        title: 图表标题
        n_samples: 用于t-SNE的样本数量（为了计算效率）
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("❌ 需要安装scikit-learn来使用t-SNE: pip install scikit-learn")
        return
    
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # 如果样本数量太多，随机采样
    if weights.shape[0] > n_samples:
        indices = np.random.choice(weights.shape[0], n_samples, replace=False)
        weights_sample = weights[indices]
        print(f"📊 从 {weights.shape[0]} 个样本中随机选择 {n_samples} 个进行t-SNE可视化")
    else:
        weights_sample = weights
        n_samples = weights.shape[0]
    
    print("🔄 正在计算t-SNE...")
    
    # 执行t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//4))
    weights_2d = tsne.fit_transform(weights_sample)
    
    plt.figure(figsize=(12, 12))
    
    # 绘制散点图
    plt.scatter(weights_2d[:, 0], weights_2d[:, 1], alpha=0.6, s=20, color='steelblue')
    
    plt.title(f'{title} (n={n_samples})', fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    plt.ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ t-SNE可视化已保存到 {save_path}")
    
    plt.close()  # 关闭图形以释放内存


def create_comprehensive_prototype_analysis(weights, save_dir, prefix="prototype_analysis"):
    """
    创建全面的原型分析可视化
    Args:
        weights: 权重数组，形状为(N, K)
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    import os
    
    print("📊 开始创建全面的原型分析可视化...")
    
    # 1. 权重分布分析（包含各原型分布和最大权重分布）
    plot_weight_distribution(
        weights,
        save_path=os.path.join(save_dir, f"{prefix}_weight_distribution_analysis.png"),
        title="Weight Distribution Analysis"
    )
    
    # 2. 平均原型使用情况
    plot_average_prototype_usage(
        weights,
        save_path=os.path.join(save_dir, f"{prefix}_prototype_usage.png"),
        title="Average Prototype Usage"
    )
    
    # 3. t-SNE可视化
    plot_tsne_sample_weights(
        weights,
        save_path=os.path.join(save_dir, f"{prefix}_tsne.png"),
        title="t-SNE of Sample Weights"
    )
    
    print("✅ 全面的原型分析可视化完成!")
