import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import matplotlib.pyplot as plt

# === Dataset ===
class TactileDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.normal_mask = []
        file_paths = sorted(glob(os.path.join(root_dir, "episode_*", "tactile.npy")))
        for path in file_paths:
            data = np.load(path)
            count = 0
            if data.ndim == 5:
                for t in range(data.shape[0]):
                    frame = data[t, -1]
                    self.samples.append(frame[0:3])
                    self.samples.append(frame[3:6])
                    self.normal_mask.append(count < 40)
                    self.normal_mask.append(count < 40)
                    count += 2
        self.samples = np.stack(self.samples)
        self.samples = (self.samples - self.samples.mean(axis=(0, 2, 3), keepdims=True)) / \
               (self.samples.std(axis=(0, 2, 3), keepdims=True) + 1e-6)
        self.normal_mask = np.array(self.normal_mask, dtype=bool)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), self.normal_mask[idx]

# === Feature Dataset ===
class TactileFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# === CNN Encoder ===
class TactileCNN(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, output_dim)
        )

    def forward(self, x):
        return self.encoder(x)

# === MLP Classifier ===
class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim=128, output_dim=11):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# === Visualization ===
class TactileVisualizer:
    def __init__(self):
        pass

    def plot_vector_map(self, vector_data, title, save_path):
        # 检查输入类型并转换为numpy数组
        if isinstance(vector_data, torch.Tensor):
            vector_data = vector_data.detach().cpu().numpy()
        # vector_data 是形状为 (3, H, W) 的张量
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.quiver(
            vector_data[0], vector_data[1], angles='xy', scale_units='xy', scale=1, color='r', label='X-axis'
        )
        ax.quiver(
            vector_data[0], vector_data[2], angles='xy', scale_units='xy', scale=1, color='b', label='Y-axis'
        )
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.legend()
        plt.grid()
        plt.savefig(save_path)
        plt.close()

# === Pipeline ===
def run_pipeline(data_root, output_dir, maxn_unnorm=5, auto_select_n=True, num_epochs=20, force_n_unnorm=None):
    os.makedirs(output_dir, exist_ok=True)
    dataset = TactileDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    cnn_model = TactileCNN(output_dim=128).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)

    all_features = []
    normal_mask_all = []
    with torch.no_grad():
        for batch, mask in dataloader:
            batch = batch.to(device)
            feats = cnn_model(batch)
            all_features.append(feats.cpu().numpy())
            normal_mask_all.append(mask.numpy())
    features = np.concatenate(all_features, axis=0)
    normal_mask = np.concatenate(normal_mask_all, axis=0)

    np.save(os.path.join(output_dir, "features.npy"), features)

    features_std = StandardScaler().fit_transform(features)
    features_pca = PCA(n_components=50).fit_transform(features_std)

    if force_n_unnorm is not None:
        maxn_unnorm = force_n_unnorm
        print(f"[Forced] Number of clusters set to: {maxn_unnorm}")
    else :
        best_n, lowest_bic = maxn_unnorm, np.inf
        for n in range(4, 7):
            gmm_tmp = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
            gmm_tmp.fit(features_pca)
            bic = gmm_tmp.bic(features_pca)
            if bic < lowest_bic:
                best_n, lowest_bic = n, bic
        maxn_unnorm = best_n
        print(f"[Auto] Best number of clusters selected by BIC: {maxn_unnorm}")

    gmm = GaussianMixture(n_components=maxn_unnorm, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(features_pca)
    probs = gmm.predict_proba(features_pca)

    multi_labels = np.zeros((len(features), maxn_unnorm + 1))
    multi_labels[:, :maxn_unnorm] = probs
    multi_labels[:, maxn_unnorm] = 0.2
    multi_labels[normal_mask, maxn_unnorm] = 1.0  # Inject prior for known normal frames

    np.save(os.path.join(output_dir, "multi_labels.npy"), multi_labels)

    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(features_pca)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar()
    plt.title("GMM Clusters (t-SNE)")
    plt.savefig(os.path.join(output_dir, "tsne_clusters.png"))

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=multi_labels[:, maxn_unnorm], cmap='Blues', alpha=0.8)
    plt.colorbar()
    plt.title("'Normal' Mode Probability Map")
    plt.savefig(os.path.join(output_dir, "grasp_softmap.png"))

    ent = entropy(probs.T)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=ent, cmap='plasma', s=6)
    plt.colorbar()
    plt.title("Uncertainty (Entropy) Map")
    plt.savefig(os.path.join(output_dir, "uncertainty_entropy_map.png"))

    # === Train multi-label classifier ===
    train_idx, test_idx = train_test_split(np.arange(len(features)), test_size=0.2, random_state=42)
    train_set = TactileFeatureDataset(features[train_idx], multi_labels[train_idx])
    test_set = TactileFeatureDataset(features[test_idx], multi_labels[test_idx])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64)

    clf = MultiLabelMLP(input_dim=features.shape[1], output_dim=multi_labels.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        clf.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = clf(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # 在分类网络预测部分添加可视化代码
    clf.eval()
    all_preds, all_labels = [], []
    visualizer = TactileVisualizer()
    vis_output_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            logits = clf(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.2).astype(int)
            
            # 获取原始触觉数据
            start_idx = batch_idx * test_loader.batch_size
            for i, pred in enumerate(preds):
                sample_idx = start_idx + i
                if sample_idx >= len(dataset):
                    break
                
                # 获取触觉数据
                tactile_data = dataset.samples[sample_idx]
                
                # 找出预测为1的类别
                pred_classes = np.where(pred == 1)[0]
                
                # 为每个预测类别生成可视化
                for pred_class in pred_classes:
                    class_dir = os.path.join(vis_output_dir, f"class_{pred_class}")
                    os.makedirs(class_dir, exist_ok=True)
                    save_path = os.path.join(class_dir, f"sample_{sample_idx}.png")
                    visualizer.plot_vector_map(
                        tactile_data, 
                        f"Class {pred_class} - Sample {sample_idx}",
                        save_path
                    )
            
            all_preds.append(preds)
            all_labels.append(y.numpy())

    y_true = (np.concatenate(all_labels) > 0.5).astype(int)  # Soft label → binary
    y_pred = np.concatenate(all_preds)

    print("\nMulti-label Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[f"cluster_{i}" for i in range(multi_labels.shape[1])], zero_division=0))

if __name__ == "__main__":
    run_pipeline(
        data_root="./organized_data/env_0",
        output_dir="./cluster/pipeline_output",
        maxn_unnorm=4,
        force_n_unnorm=3,
        num_epochs=100
    )
