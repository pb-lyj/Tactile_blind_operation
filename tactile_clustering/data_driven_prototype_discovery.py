import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from Physical_mapping import save_physicalXYZ_images

# ==================== Config ====================
DATA_ROOT = "./organized_data/env_0"
OUTPUT_DIR = "./cluster/prototype_library"
NUM_PROTOTYPES = 64
BATCH_SIZE = 128
EPOCHS = 100
LR = 1e-3

# ==================== Dataset ====================
class TactileSampleDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        paths = sorted(glob(os.path.join(root_dir, "episode_*", "tactile.npy")))
        for path in paths:
            data = np.load(path)  # shape (T, 6, H, W)
            for t in range(data.shape[0]):
                frame = data[t, -1]  # use final frame
                self.samples.append(frame[0:3])  # sensor 1
                self.samples.append(frame[3:6])  # sensor 2
        self.samples = np.stack(self.samples)
        self.samples[:, 0:2] /= 0.1  # normalize XY range (empirically ~[-0.2, 0.2])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)

# ==================== Model ====================
class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Localization network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # 2x3 affine matrix
        )
        # Initialize the weights/bias with identity transformation
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.localization(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        transformed = F.grid_sample(x, grid, align_corners=True)
        return transformed, theta

class PrototypeAutoencoder(nn.Module):
    def __init__(self, num_prototypes=64, input_shape=(3, 32, 32)):
        super().__init__()
        self.K = num_prototypes
        self.H, self.W = input_shape[1:]

        # 保持原有的原型库不变
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, *input_shape))

        # 保持原有的编码器不变
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_prototypes),
            nn.Softmax(dim=-1)
        )

        # 添加 STN
        self.stn = SpatialTransformer()

    def forward(self, x):
        # 编码器输出权重
        weights = self.encoder(x)  # (B, K)
        
        # 原型组合
        proto_combined = torch.einsum('bk,kchw->bchw', weights, self.prototypes)
        
        # STN 变换
        transformed, theta = self.stn(proto_combined)
        
        return transformed, weights, proto_combined, theta

# ==================== Training ====================
def compute_transform_consistency_loss(prototypes):
    """计算原型之间的变换一致性损失"""
    K, C, H, W = prototypes.shape
    protos_flat = prototypes.view(K, -1)
    
    # 归一化原型
    protos_norm = F.normalize(protos_flat, dim=1)
    
    # 计算变换一致性损失
    loss = 0
    for i in range(K):
        for j in range(i+1, K):
            pi = protos_flat[i]
            pj = protos_flat[j]
            
            # 缩放一致性
            scale_ratio = torch.norm(pi) / (torch.norm(pj) + 1e-6)
            scale_loss = torch.abs(scale_ratio - 1.0)
            
            # 旋转一致性
            cos_sim = torch.sum(protos_norm[i] * protos_norm[j])
            rotation_loss = cos_sim.abs()
            
            loss += scale_loss + rotation_loss
            
    return loss / (K * (K-1) / 2)

def train_prototypes():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dataset = TactileSampleDataset(DATA_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    model = PrototypeAutoencoder(NUM_PROTOTYPES).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            batch = batch.cuda()
            
            # 前向传播
            transformed, weights, proto_combined, theta = model(batch)
            
            # 1. 重构损失（在STN变换后计算）
            recon_loss = F.mse_loss(transformed, batch)
            
            # 2. 原型多样性损失（在CNN后计算）
            norm_prototypes = F.normalize(model.prototypes.view(NUM_PROTOTYPES, -1), dim=1)
            similarity = torch.matmul(norm_prototypes, norm_prototypes.T)
            diversity_loss = (similarity**2).mean() - (1.0 / NUM_PROTOTYPES)
            
            # 3. 变换一致性损失（在CNN后计算）
            transform_loss = compute_transform_consistency_loss(model.prototypes)
            
            # 组合损失
            loss = recon_loss + 0.05 * diversity_loss + 0.1 * transform_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.size(0)
        
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")

    # Save prototypes
    prototypes_np = model.prototypes.detach().cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "prototypes.npy"), prototypes_np)
    # prototypes physical visualization
    save_physicalXYZ_images(prototypes_np, OUTPUT_DIR)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "prototype_model.pt"))

    # Save sample weights and t-SNE + metrics
    save_sample_weights(model, dataset)

# ==================== Sample Weights + t-SNE + Metrics ====================
def save_sample_weights(model, dataset):
    model.eval()
    loader = DataLoader(dataset, batch_size=256)
    all_weights = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.cuda()
            _, weights, _, _ = model(batch)  # transformed, weights, proto_combined, theta
            all_weights.append(weights.cpu().numpy())

    weights_np = np.concatenate(all_weights, axis=0)
    np.save(os.path.join(OUTPUT_DIR, "sample_weights.npy"), weights_np)

    # ----- t-SNE -----
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_embed = tsne.fit_transform(weights_np)
    plt.figure(figsize=(6, 6))
    plt.scatter(tsne_embed[:, 0], tsne_embed[:, 1], s=2, alpha=0.6)
    plt.title("t-SNE of Sample Weights")
    plt.savefig(os.path.join(OUTPUT_DIR, "tsne_weights.png"))
    plt.close()


if __name__ == '__main__':
    train_prototypes()
