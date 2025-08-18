import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.logging import Logger
from utils.visualization import save_physicalXYZ_images, plot_loss_curves
from utils.data_utils import save_sample_weights_and_analysis
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.manifold import TSNE
from models.prototype_ae_STN import PrototypeAutoencoder, compute_losses
from models.prototype_ae_baseline import PrototypeAEBaseline, compute_baseline_losses
from models.improved_prototype_ae import ImprovedForcePrototypeAE, compute_improved_losses
from models.improved_prototype_ae_STN import ImprovedPrototypeSTNAE, compute_improved_stn_losses

# ==================== Config ====================
# 新的数据根目录 - data25.7_aligned中的各类别
DATA_CATEGORIES = [
    "cir_lar", "cir_med", "cir_sma",
    "rect_lar", "rect_med", "rect_sma", 
    "tri_lar", "tri_med", "tri_sma"
]
DATA_ROOT = "./data/data25.7_aligned"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join("./cluster/prototype_library", timestamp)
NUM_PROTOTYPES = 8  # 原型数量
BATCH_SIZE = 64     # 减小批次大小，提高梯度稳定性
EPOCHS = 50        # 增加训练轮数
LR = 1e-4          # 降低学习率
START_FRAME = 0    # 从第几帧开始截取

# ==================== Dataset ====================

class TactileForcesDataset(Dataset):
    def __init__(self, data_root, categories=None, start_frame=START_FRAME, exclude_test_folders=True):
        """
        Args:
            data_root: 数据根目录路径 (data25.7_aligned)
            categories: 要包含的类别列表，如 ["cir_lar", "rect_med"] 等
            start_frame: 从第几帧开始截取数据
            exclude_test_folders: 是否排除测试文件夹(第10,20,30,40,50个)
        """
        self.samples = []
        if categories is None:
            categories = DATA_CATEGORIES
            
        # 定义要排除的测试文件夹索引 (1-based)
        test_folder_indices = {10, 20, 30, 40, 50} if exclude_test_folders else set()
        
        total_frames = 0
        valid_frames = 0
        total_episodes = 0
        excluded_episodes = 0
        
        for category in categories:
            category_path = os.path.join(data_root, category)
            if not os.path.exists(category_path):
                print(f"警告: 类别路径不存在: {category_path}")
                continue
                
            # 获取该类别下的所有样本目录，按名称排序确保顺序一致
            sample_dirs = sorted(glob(os.path.join(category_path, "2025*")))
            
            for idx, sample_dir in enumerate(sample_dirs, 1):  # 1-based索引
                total_episodes += 1
                
                # 检查是否是测试文件夹
                if idx in test_folder_indices:
                    excluded_episodes += 1
                    # print(f"排除测试文件夹: {category} 第{idx}个 - {os.path.basename(sample_dir)}")
                    continue
                
                try:
                    # 加载左右手的力数据
                    forces_l_path = os.path.join(sample_dir, "_forces_l.npy")
                    forces_r_path = os.path.join(sample_dir, "_forces_r.npy")
                    
                    if not (os.path.exists(forces_l_path) and os.path.exists(forces_r_path)):
                        continue
                        
                    forces_l = np.load(forces_l_path)  # shape (T, 3, 20, 20)
                    forces_r = np.load(forces_r_path)  # shape (T, 3, 20, 20)
                    
                    # 确保左右手数据长度一致
                    min_length = min(forces_l.shape[0], forces_r.shape[0])
                    forces_l = forces_l[:min_length]
                    forces_r = forces_r[:min_length]
                    
                    total_frames += min_length
                    
                    # 只使用从 start_frame 开始的数据
                    if min_length > start_frame:
                        for t in range(start_frame, min_length):
                            # 取每个时间步的数据
                            frame_l = forces_l[t]  # shape (3, 20, 20) - 左手传感器
                            frame_r = forces_r[t]  # shape (3, 20, 20) - 右手传感器
                            
                            self.samples.append(frame_l)  # 左手数据
                            self.samples.append(frame_r)  # 右手数据
                            valid_frames += 1
                            
                except Exception as e:
                    print(f"处理样本 {sample_dir} 时出错: {e}")
                    continue
            print(f"共排除 {category}类 测试文件夹 {excluded_episodes}个")
                    
        if len(self.samples) == 0:
            raise ValueError("没有找到有效的数据样本！")
            
        self.samples = np.stack(self.samples)
        
        # 数据归一化 - 根据实际的力数据范围进行调整
        # 这里可能需要根据实际数据分布调整归一化策略
        # 暂时使用简单的标准化
        self.samples = self._normalize_data(self.samples)
        
        print(f"[TactileForcesDataset] 数据统计:")
        print(f"  - 数据根目录: {data_root}")
        print(f"  - 包含类别: {categories}")
        print(f"  - 总情节数: {total_episodes}")
        if exclude_test_folders:
            print(f"  - 排除测试情节数: {excluded_episodes} (第10,20,30,40,50个文件夹)")
            print(f"  - 训练情节数: {total_episodes - excluded_episodes}")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 截取起始帧: {start_frame}")
        print(f"  - 有效帧数: {valid_frames}")
        print(f"  - 总样本数: {len(self.samples)} (包含左右手传感器)")
        print(f"  - 样本形状: {self.samples.shape}")
        print(f"  - 数据范围: [{self.samples.min():.4f}, {self.samples.max():.4f}]")

    def _normalize_data(self, data):
        """
        改进的数据归一化处理
        Args:
            data: 原始数据 (N, 3, 20, 20)
        Returns:
            标准化后的数据
        """
        print(f"原始数据范围: [{data.min():.4f}, {data.max():.4f}]")
        
        # 方法3: 分位数归一化 + Z-score标准化（推荐）
        normalized_data = np.zeros_like(data)
        
        for i in range(data.shape[1]):  # 对每个通道分别处理
            channel_data = data[:, i]
            
            # 1. 使用分位数去除极端异常值
            q1, q99 = np.percentile(channel_data, [1, 99])
            channel_data = np.clip(channel_data, q1, q99)
            
            # 2. Z-score标准化
            mean = channel_data.mean()
            std = channel_data.std()
            if std > 1e-8:
                normalized_data[:, i] = (channel_data - mean) / std
            else:
                normalized_data[:, i] = channel_data - mean
                
        print(f"标准化后数据范围: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
        
        # 检查是否有NaN或Inf
        if np.isnan(normalized_data).any() or np.isinf(normalized_data).any():
            print("警告: 标准化后数据包含NaN或Inf值")
            normalized_data = np.nan_to_num(normalized_data, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return normalized_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)

# ==================== Training ====================

def get_model_config(model_type="ImprovedPrototypeSTNAE"):
    """
    获取不同模型的配置信息
    Args:
        model_type: 模型类型 ["PrototypeAutoencoder", "PrototypeAEBaseline", 
                    "ImprovedForcePrototypeAE", "ImprovedPrototypeSTNAE"]
    Returns:
        dict: 包含模型、损失函数、损失字段等配置
    """
    if model_type == "PrototypeAutoencoder":
        from models.prototype_ae_STN import PrototypeAutoencoder, compute_losses
        model = PrototypeAutoencoder(NUM_PROTOTYPES, input_shape=(3, 20, 20), share_stn=True).cuda()
        compute_loss_fn = compute_losses
        loss_fields = ['recon_loss', 'diversity_loss', 'entropy_loss', 'stn_loss']
        title_suffix = "Original STN Prototype AE"
        
    elif model_type == "PrototypeAEBaseline":
        from models.prototype_ae_baseline import PrototypeAEBaseline, compute_baseline_losses
        model = PrototypeAEBaseline(NUM_PROTOTYPES, input_shape=(3, 20, 20)).cuda()
        compute_loss_fn = compute_baseline_losses
        loss_fields = ['recon_loss', 'diversity_loss', 'sparsity_loss']
        title_suffix = "Baseline Prototype AE"
        
    elif model_type == "ImprovedForcePrototypeAE":
        from models.improved_prototype_ae import ImprovedForcePrototypeAE, compute_improved_losses
        model = ImprovedForcePrototypeAE(NUM_PROTOTYPES, input_shape=(3, 20, 20)).cuda()
        compute_loss_fn = compute_improved_losses
        loss_fields = ['recon_loss', 'diversity_loss', 'entropy_loss', 'sparsity_loss', 'gini_coeff']
        title_suffix = "Improved Force Prototype AE"
        
    elif model_type == "ImprovedPrototypeSTNAE":
        from models.improved_prototype_ae_STN import ImprovedPrototypeSTNAE, compute_improved_stn_losses
        model = ImprovedPrototypeSTNAE(
            num_prototypes=NUM_PROTOTYPES, 
            input_shape=(3, 20, 20),
            share_stn=False
        ).cuda()
        compute_loss_fn = compute_improved_stn_losses
        loss_fields = ['recon_loss', 'diversity_loss', 'entropy_loss', 'sparsity_loss', 'gini_coeff', 'stn_loss', 'theta_diversity_loss']
        title_suffix = "Improved STN Prototype AE"
        
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return {
        'model': model,
        'compute_loss_fn': compute_loss_fn,
        'loss_fields': loss_fields,
        'title_suffix': title_suffix,
        'model_type': model_type
    }

def train_prototypes(model_type="ImprovedPrototypeSTNAE", progressive_stn=False):
    dataset = TactileForcesDataset(DATA_ROOT, DATA_CATEGORIES)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # 获取模型配置
    config = get_model_config(model_type)
    model = config['model']
    compute_loss_fn = config['compute_loss_fn']
    loss_fields = config['loss_fields']
    title_suffix = config['title_suffix']
    
    print(f"使用模型: {model_type}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 渐进式STN训练策略
    if progressive_stn and model_type == "ImprovedPrototypeSTNAE" and not model.share_stn:
        print(f"🔄 启用渐进式STN训练策略:")
        print(f"  - 前 {int(EPOCHS * 0.6)} 个epoch: 同步更新所有STN参数")
        print(f"  - 后 {int(EPOCHS * 0.4)} 个epoch: 独立更新各STN参数")
        return train_prototypes_progressive_stn(model, compute_loss_fn, loss_fields, title_suffix, dataset, loader)
    
    # 使用AdamW优化器，带权重衰减
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # 学习率调度器 - 移除verbose参数以兼容不同PyTorch版本
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_loss = float('inf')
    patience_counter = 0
    patience = 20  # 早停patience
    
    # 记录损失历史用于绘制曲线 - 动态适配不同模型的损失字段
    loss_history = {'epoch': [], 'total_loss': []}
    for field in loss_fields:
        loss_history[field] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        # 动态初始化指标累积器
        metrics_sum = {field: 0 for field in loss_fields}
        
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            batch = batch.cuda()
            
            # 前向传播 - 适配不同模型的输出
            if model_type in ["ImprovedPrototypeSTNAE", "PrototypeAutoencoder"]:
                # STN模型返回4个值
                recon, weights, transformed_protos, thetas = model(batch)
                
                # 根据模型类型选择损失函数参数
                if model_type == "ImprovedPrototypeSTNAE":
                    loss, metrics = compute_loss_fn(
                        batch, recon, weights, transformed_protos, thetas,
                        diversity_lambda=0.5, entropy_lambda=0.1, 
                        sparsity_lambda=0.03, stn_reg_lambda=0.01
                    )
                else:  # PrototypeAutoencoder
                    loss, metrics = compute_loss_fn(
                        batch, recon, weights, transformed_protos, thetas,
                        diversity_lambda=0.1, entropy_lambda=10.0, stn_reg_lambda=0.05
                    )
            else:
                # 非STN模型返回3个值
                recon, weights, protos = model(batch)
                
                if model_type == "ImprovedForcePrototypeAE":
                    loss, metrics = compute_loss_fn(
                        batch, recon, weights, protos,
                        diversity_lambda=1.0, entropy_lambda=0.1, sparsity_lambda=0.01
                    )
                else:  # PrototypeAEBaseline
                    loss, metrics = compute_loss_fn(
                        batch, recon, weights, protos,
                        diversity_lambda=0.1, sparsity_lambda=0.01
                    )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 累积损失和指标
            batch_size = batch.size(0)
            total_loss += loss.item() * batch_size
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v * batch_size
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(dataset)
        avg_metrics = {k: v/len(dataset) for k, v in metrics_sum.items()}
        
        # 学习率调度
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 手动记录学习率变化
        if current_lr != prev_lr:
            print(f"学习率从 {prev_lr:.2e} 降到 {current_lr:.2e}")
        
        # 打印训练信息
        print(f"Epoch {epoch} Loss: {avg_loss:.4f} LR: {current_lr:.2e}")
        print("Metrics:", " ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items()))
        
        # 记录损失历史
        loss_history['epoch'].append(epoch)
        loss_history['total_loss'].append(avg_loss)
        for key, value in avg_metrics.items():
            if key in loss_history:
                loss_history[key].append(value)
        
        # 早停检查
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            # 保存最好的模型
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # 保存模型和原型
    prototypes_np = model.prototypes.detach().cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "prototypes.npy"), prototypes_np)
    save_physicalXYZ_images(prototypes_np, OUTPUT_DIR)  # 使用utils统一函数
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "prototype_model.pt"))

    # 绘制并保存训练损失曲线
    plot_loss_curves(
        loss_history, 
        save_path=os.path.join(OUTPUT_DIR, "training_loss_curves.png"),
        title=f"{title_suffix} Training Loss"
    )
    
    # 保存损失历史数据
    np.save(os.path.join(OUTPUT_DIR, "loss_history.npy"), loss_history)

    # 保存样本权重和可视化
    # 保存样本权重并生成分析图表
    save_sample_weights_and_analysis(model, dataset, output_dir=OUTPUT_DIR)


def train_prototypes_progressive_stn(model, compute_loss_fn, loss_fields, title_suffix, dataset, loader):
    """
    渐进式STN训练策略：前40%同步，后60%独立
    """
    print("🚀 开始渐进式STN训练...")
    
    # 计算阶段划分
    sync_epochs = int(EPOCHS * 0.6)
    independent_epochs = EPOCHS - sync_epochs
    
    # 记录损失历史
    loss_history = {'epoch': [], 'total_loss': []}
    for field in loss_fields:
        loss_history[field] = []
    
    best_loss = float('inf')
    
    # ==================== 阶段1: 同步STN训练 (前40%) ====================
    print(f"\n📍 阶段1: 同步STN训练 (1-{sync_epochs} epochs)")
    print("所有STN模块共享梯度更新...")
    
    # 阶段1优化器 - 稍高学习率
    optimizer_sync = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler_sync = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_sync, mode='min', factor=0.7, patience=8
    )
    
    for epoch in range(1, sync_epochs + 1):
        model.train()
        total_loss = 0
        metrics_sum = {field: 0 for field in loss_fields}
        
        # 同步更新策略：让所有STN模块的梯度保持一致
        for batch in tqdm(loader, desc=f"同步训练 Epoch {epoch}/{sync_epochs}"):
            batch = batch.cuda()
            
            # 前向传播
            recon, weights, transformed_protos, thetas = model(batch)
            
            # 计算损失
            loss, metrics = compute_loss_fn(
                batch, recon, weights, transformed_protos, thetas,
                diversity_lambda=0.5, entropy_lambda=0.1, 
                sparsity_lambda=0.03, stn_reg_lambda=0.01
            )
            
            # 反向传播
            optimizer_sync.zero_grad()
            loss.backward()
            
            # 🔑 关键：同步所有STN模块的梯度
            if hasattr(model, 'stn_modules'):
                sync_stn_gradients(model.stn_modules)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_sync.step()
            
            total_loss += loss.item()
            for field in loss_fields:
                if field in metrics:
                    metrics_sum[field] += metrics[field]
        
        # 记录本epoch结果
        avg_loss = total_loss / len(loader)
        avg_metrics = {field: metrics_sum[field] / len(loader) for field in loss_fields}
        
        loss_history['epoch'].append(epoch)
        loss_history['total_loss'].append(avg_loss)
        for field in loss_fields:
            loss_history[field].append(avg_metrics[field])
        
        scheduler_sync.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model_sync.pt"))
        
        if epoch % 10 == 0:
            print(f'同步阶段 Epoch {epoch}, Loss: {avg_loss:.4f}')
            print(f'  主要指标: recon={avg_metrics.get("recon_loss", 0):.4f}, '
                  f'stn={avg_metrics.get("stn_loss", 0):.4f}')
    
    print(f"✅ 阶段1完成！最佳损失: {best_loss:.4f}")
    
    # ==================== 阶段2: 独立STN训练 (后60%) ====================
    print(f"\n📍 阶段2: 独立STN训练 ({sync_epochs+1}-{EPOCHS} epochs)")
    print("各STN模块独立更新，学习特定变换...")
    
    # 阶段2优化器 - 降低学习率，更精细调节
    optimizer_independent = torch.optim.AdamW(model.parameters(), lr=LR*0.5, weight_decay=1e-4)
    scheduler_independent = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_independent, mode='min', factor=0.8, patience=10
    )
    
    for epoch in range(sync_epochs + 1, EPOCHS + 1):
        model.train()
        total_loss = 0
        metrics_sum = {field: 0 for field in loss_fields}
        
        for batch in tqdm(loader, desc=f"独立训练 Epoch {epoch}/{EPOCHS}"):
            batch = batch.cuda()
            
            # 前向传播
            recon, weights, transformed_protos, thetas = model(batch)
            
            # 计算损失 - 增加theta多样性权重
            loss, metrics = compute_loss_fn(
                batch, recon, weights, transformed_protos, thetas,
                diversity_lambda=0.5, entropy_lambda=0.1, 
                sparsity_lambda=0.03, stn_reg_lambda=0.008  # 稍微降低STN正则化
            )
            
            # 反向传播 - 正常独立更新
            optimizer_independent.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_independent.step()
            
            total_loss += loss.item()
            for field in loss_fields:
                if field in metrics:
                    metrics_sum[field] += metrics[field]
        
        # 记录本epoch结果
        avg_loss = total_loss / len(loader)
        avg_metrics = {field: metrics_sum[field] / len(loader) for field in loss_fields}
        
        loss_history['epoch'].append(epoch)
        loss_history['total_loss'].append(avg_loss)
        for field in loss_fields:
            loss_history[field].append(avg_metrics[field])
        
        scheduler_independent.step(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        
        if epoch % 10 == 0:
            print(f'独立阶段 Epoch {epoch}, Loss: {avg_loss:.4f}')
            print(f'  主要指标: recon={avg_metrics.get("recon_loss", 0):.4f}, '
                  f'theta_div={avg_metrics.get("theta_diversity_loss", 0):.4f}')
    
    print(f"✅ 渐进式训练完成！最终最佳损失: {best_loss:.4f}")
    
    # 保存结果
    prototypes_np = model.prototypes.detach().cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "prototypes.npy"), prototypes_np)
    save_physicalXYZ_images(prototypes_np, OUTPUT_DIR)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "prototype_model.pt"))
    
    # 绘制训练曲线
    plot_loss_curves(
        loss_history, 
        save_path=os.path.join(OUTPUT_DIR, "training_loss_curves.png"),
        title=f"{title_suffix} Progressive Training Loss"
    )
    
    # 添加阶段分界线标记
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    epochs = loss_history['epoch']
    plt.plot(epochs, loss_history['total_loss'], 'b-', linewidth=2, label='Total Loss')
    plt.axvline(x=sync_epochs, color='red', linestyle='--', alpha=0.7, 
                label=f'Phase Switch (Epoch {sync_epochs})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title_suffix} Progressive Training - Phase Transition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "progressive_training_phases.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    np.save(os.path.join(OUTPUT_DIR, "loss_history.npy"), loss_history)
    save_sample_weights_and_analysis(model, dataset, output_dir=OUTPUT_DIR)
    
    return model, loss_history


def sync_stn_gradients(stn_modules):
    """
    同步多个STN模块的梯度，让它们学习相似的变换
    """
    if len(stn_modules) <= 1:
        return
    
    # 计算所有STN模块参数的平均梯度
    avg_grads = {}
    
    # 收集所有梯度
    for name, param in stn_modules[0].named_parameters():
        if param.grad is not None:
            grad_sum = param.grad.clone()
            count = 1
            
            # 累加其他模块的对应梯度
            for i in range(1, len(stn_modules)):
                other_param = dict(stn_modules[i].named_parameters())[name]
                if other_param.grad is not None:
                    grad_sum += other_param.grad
                    count += 1
            
            avg_grads[name] = grad_sum / count
    
    # 将平均梯度分配给所有模块
    for stn_module in stn_modules:
        for name, param in stn_module.named_parameters():
            if name in avg_grads and param.grad is not None:
                param.grad.copy_(avg_grads[name])


if __name__ == '__main__':
    # 可以通过命令行参数或环境变量选择模型类型和训练策略
    import sys
    
    # 默认参数
    model_type = "ImprovedPrototypeSTNAE"
    progressive_stn = False
    
    # 解析命令行参数
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2].lower() in ['true', '1', 'progressive']:
        progressive_stn = True
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_suffix = "_progressive" if progressive_stn else ""
    sys.stdout = Logger(f"{OUTPUT_DIR}/train_forces_{model_type.lower()}{log_suffix}.log")
    
    # ======================= Report =======================
    print("=" * 60)
    print(f"Forces Prototype Discovery Configuration:")
    print(f"MODEL TYPE       = {model_type}")
    if model_type == "ImprovedPrototypeSTNAE":
        print(f"SHARE_STN        = False (独立STN)")
        if progressive_stn:
            print(f"TRAINING MODE    = Progressive (前40%同步，后60%独立)")
        else:
            print(f"TRAINING MODE    = Standard (全程独立)")
    print(f"DATA_ROOT        = {DATA_ROOT}")
    print("DATA_CATEGORIES:")
    for category in DATA_CATEGORIES:
        print(f"  - {category}")
    print(f"OUTPUT_DIR       = {OUTPUT_DIR}")
    print(f"NUM_PROTOTYPES   = {NUM_PROTOTYPES}")
    print(f"BATCH_SIZE       = {BATCH_SIZE}")
    print(f"EPOCHS           = {EPOCHS}")
    print(f"LR               = {LR}")
    print(f"START_FRAME      = {START_FRAME}")
    if progressive_stn:
        print(f"SYNC_EPOCHS      = {int(EPOCHS * 0.4)} (40%)")
        print(f"INDEPENDENT_EPOCHS = {int(EPOCHS * 0.6)} (60%)")
    print("=" * 60)
    
    train_prototypes(model_type=model_type, progressive_stn=progressive_stn)
