import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from Physical_mapping import Logger
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.manifold import TSNE
from Physical_mapping import save_physicalXYZ_images
from models.prototype_ae import PrototypeAutoencoder, compute_losses
from models.prototype_ae_baseline import PrototypeAEBaseline, compute_baseline_losses

# ==================== Config ====================
DATA_ROOTS = [
    "./organized_data_1_1/env_0",
    "./organized_data_1_1/env_1",
    "./organized_data_2_1/env_0",
    "./organized_data_2_1/env_1",
    "./organized_data_3_1/env_0",
    "./organized_data_3_1/env_1"
]
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join("./cluster/prototype_library", timestamp)
NUM_PROTOTYPES = 4
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
START_FRAME = 20  # 添加配置：从第几帧开始截取

# ==================== Dataset ====================
class TactileSampleDataset(Dataset):
    def __init__(self, root_dirs, start_frame=START_FRAME):
        """
        Args:
            root_dirs: 字符串或字符串列表，数据目录路径
            start_frame: 从第几帧开始截取数据
        """
        self.samples = []
        if isinstance(root_dirs, str):
            root_dirs = [root_dirs]
            
        total_frames = 0
        valid_frames = 0
        
        for root_dir in root_dirs:
            paths = sorted(glob(os.path.join(root_dir, "episode_*", "tactile.npy")))
            for path in paths:
                data = np.load(path)  # shape (T, 6, H, W)
                total_frames += data.shape[0]
                
                # 只使用从 start_frame 开始的数据
                if data.shape[0] > start_frame:
                    for t in range(start_frame, data.shape[0]):
                        frame = data[t, -1]  # use final frame
                        self.samples.append(frame[0:3])  # sensor 1
                        self.samples.append(frame[3:6])  # sensor 2
                        valid_frames += 1
                    
        self.samples = np.stack(self.samples)
        self.samples[:, 0:2] /= 0.05  # normalize XY range
        
        total_episodes = sum(len(glob(os.path.join(root, "episode_*"))) 
                           for root in root_dirs)
        print(f"[TactileSampleDataset] 数据统计:")
        print(f"  - 环境数量: {len(root_dirs)}")
        print(f"  - 情节数量: {total_episodes}")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 截取起始帧: {start_frame}")
        print(f"  - 有效帧数: {valid_frames}")
        print(f"  - 总样本数: {len(self.samples)} (包含左右传感器)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)

# ==================== Training ====================
def train_prototypes():
    dataset = TactileSampleDataset(DATA_ROOTS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    # model = PrototypeAutoencoder(NUM_PROTOTYPES).cuda()
    model = PrototypeAEBaseline(NUM_PROTOTYPES).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        metrics_sum = {'recon_loss': 0, 'diversity_loss': 0, 'entropy_loss': 0}
        
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            batch = batch.cuda()
            
            # 前向传播
            # recon, weights, transformed_protos, thetas = model(batch)
            recon, weights, transformed_protos = model(batch)
            
            # 计算损失
            # loss, metrics = compute_losses(
            #     batch, recon, weights, 
            #     transformed_protos, thetas,
            # )
            loss, metrics = compute_baseline_losses(
                batch, recon, weights, 
                transformed_protos
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 累积损失和指标
            batch_size = batch.size(0)
            total_loss += loss.item() * batch_size
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v * batch_size
        
        # 计算平均损失和指标
        avg_loss = total_loss / len(dataset)
        avg_metrics = {k: v/len(dataset) for k, v in metrics_sum.items()}
        
        # 打印训练信息
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
        print("Metrics:", " ".join(f"{k}: {v:.4f}" for k, v in avg_metrics.items()))

    # 保存模型和原型
    prototypes_np = model.prototypes.detach().cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "prototypes.npy"), prototypes_np)
    save_physicalXYZ_images(prototypes_np, OUTPUT_DIR)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "prototype_model.pt"))

    # 保存样本权重和可视化
    save_sample_weights(model, dataset)

# ==================== Sample Weights + t-SNE ====================
def save_sample_weights(model, dataset, output_dir="./output"):
    import os
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    model.eval()
    loader = DataLoader(dataset, batch_size=256)
    all_weights = []

    os.makedirs(output_dir, exist_ok=True)

    print("收集样本权重...")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.cuda()
            # _, weights, _, _ = model(batch)
            _, weights, _ = model(batch)
            weights_cpu = weights.cpu().float().numpy()
            weights_cpu = np.nan_to_num(weights_cpu, nan=0.0, posinf=1.0, neginf=-1.0)
            all_weights.append(weights_cpu)

    weights_np = np.concatenate(all_weights, axis=0)

    print("保存权重数据...")
    np.save(os.path.join(output_dir, "sample_weights.npy"), weights_np)

    print("生成t-SNE可视化...")
    try:
        max_samples = 5000
        if len(weights_np) > max_samples:
            indices = np.random.choice(len(weights_np), max_samples, replace=False)
            weights_for_tsne = weights_np[indices]
        else:
            weights_for_tsne = weights_np

        if np.allclose(weights_for_tsne, weights_for_tsne[0]):
            raise ValueError("⚠️ 所有样本权重完全相同，跳过 t-SNE")

        # 加微扰避免 std=0
        weights_for_tsne += 1e-4 * np.random.randn(*weights_for_tsne.shape)

        # 标准化
        mean = weights_for_tsne.mean(axis=0)
        std = weights_for_tsne.std(axis=0)
        std[std == 0] = 1.0
        weights_for_tsne = (weights_for_tsne - mean) / std

        # t-SNE（用 CPU，关闭多线程）
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(weights_for_tsne) - 1),
            random_state=42,
            n_iter=1000,
            init='random',
            method='barnes_hut'
        )
        tsne_embed = tsne.fit_transform(weights_for_tsne)

        # 绘图
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], s=2, alpha=0.6)
        plt.title(f"t-SNE of Sample Weights (n={len(weights_for_tsne)})")
        plt.savefig(os.path.join(output_dir, "tsne_weights.png"), dpi=300)
        plt.close()

    except Exception as e:
        print(f"⚠️ t-SNE 可视化失败: {e}")
        import traceback
        traceback.print_exc()

    print("生成权重分布图...")
    try:
        max_vals = weights_np.max(axis=1)
        plt.figure(figsize=(16, 12))
        data_range = max_vals.max() - max_vals.min()
        if data_range > 0:
            n_bins = min(50, int(np.ceil(data_range * 100)))
            plt.hist(max_vals, bins=n_bins)
        else:
            plt.hist(max_vals, bins=10)
        plt.title("Max Weight Distribution")
        plt.xlabel("Max Activation (Sigmoid)")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, "max_weight_hist.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ 权重分布图生成失败: {e}")

    print("生成原型使用频率图...")
    try:
        mean_usage = weights_np.mean(axis=0)
        plt.figure(figsize=(24, 12))
        plt.bar(range(len(mean_usage)), mean_usage)
        plt.title("Average Prototype Usage")
        plt.xlabel("Prototype Index")
        plt.ylabel("Average Activation")
        plt.savefig(os.path.join(output_dir, "prototype_usage.png"))
        plt.close()
    except Exception as e:
        print(f"⚠️ 原型使用频率图生成失败: {e}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.stdout = Logger(f"{OUTPUT_DIR}/train_ae.log")
    
    # ======================= Report =======================
    print("=" * 50)
    print("Configuration:")
    print("DATA_ROOTS:")
    for root in DATA_ROOTS:
        print(f"  - {root}")
    print(f"OUTPUT_DIR       = {OUTPUT_DIR}")
    print(f"NUM_PROTOTYPES   = {NUM_PROTOTYPES}")
    print(f"BATCH_SIZE       = {BATCH_SIZE}")
    print(f"EPOCHS           = {EPOCHS}")
    print(f"LR               = {LR}")
    print("=" * 50)
    
    train_prototypes()
