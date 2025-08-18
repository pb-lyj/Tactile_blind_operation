import os
import sys
from datetime import datetime

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from utils.logging import Logger
from utils.visualization import save_physicalXYZ_images
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
    save_sample_weights_and_analysis(model, dataset, output_dir=OUTPUT_DIR)

# ==================== Sample Weights + t-SNE ====================
# 这个函数已迁移到 utils.data_utils.save_sample_weights_and_analysis
# ==================== Sample Weights + t-SNE ====================
# 这个函数已迁移到 utils.data_utils.save_sample_weights_and_analysis


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
