import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
                    
                    self.normal_mask.append(count < 40)  # for img1
                    self.normal_mask.append(count < 40)  # for img2
                    count += 2
        self.samples = np.stack(self.samples)
        self.normal_mask = np.array(self.normal_mask, dtype=bool)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32), self.normal_mask[idx]

# === Custom CNN Encoder ===
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

# === Main ===
def run_tactilecnn_with_normal(data_root="./env_0", output_dir="./cluster/tactilecnn_output", n_clusters=5, auto_select_n=False):
    os.makedirs(output_dir, exist_ok=True)

    dataset = TactileDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = TactileCNN(output_dim=128).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_features = []
    normal_mask_all = []
    with torch.no_grad():
        for batch, mask in dataloader:
            batch = batch.to(device)
            feats = model(batch)
            all_features.append(feats.cpu().numpy())
            normal_mask_all.append(mask.numpy())
    features = np.concatenate(all_features, axis=0)
    normal_mask = np.concatenate(normal_mask_all, axis=0)

    features = StandardScaler().fit_transform(features)
    features_pca = PCA(n_components=50).fit_transform(features)

    if auto_select_n:
        lowest_bic = np.inf
        best_n = n_clusters
        for n in range(2, 11):
            gmm_tmp = GaussianMixture(n_components=n, covariance_type='full', random_state=42)
            gmm_tmp.fit(features_pca)
            bic = gmm_tmp.bic(features_pca)
            if bic < lowest_bic:
                lowest_bic = bic
                best_n = n
        n_clusters = best_n
        print(f"[Auto] Best number of clusters selected by BIC: {n_clusters}")

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(features_pca)
    probs = gmm.predict_proba(features_pca)

    # Build multi-label matrix with additional 'normal' class
    multi_labels = np.zeros((len(features), n_clusters + 1))
    multi_labels[:, :n_clusters] = probs
    multi_labels[:, n_clusters] = normal_mask.astype(float)

    np.save(os.path.join(output_dir, "multi_labels.npy"), multi_labels)
    np.save(os.path.join(output_dir, "gmm_labels.npy"), labels)
    np.save(os.path.join(output_dir, "gmm_probs.npy"), probs)

    sil_score = silhouette_score(features_pca, labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    dominant_cluster = np.argmax((probs > 0.3).sum(axis=0))
    print(f"Identified dominant 'grasp' cluster: {dominant_cluster}")

    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(features_pca)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("TactileCNN GMM Clusters (t-SNE)")
    plt.savefig(os.path.join(output_dir, "tactilecnn_gmm_tsne.png"))

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=multi_labels[:, n_clusters], cmap='Blues', alpha=0.8)
    plt.colorbar()
    plt.title("Explicit 'Normal' Mode (Prior Injected) Probability Map")
    plt.savefig(os.path.join(output_dir, "grasp_softmap.png"))

    ent = entropy(probs.T)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=ent, cmap='plasma', s=6)
    plt.colorbar()
    plt.title("Uncertainty (Entropy) Map")
    plt.savefig(os.path.join(output_dir, "uncertainty_entropy_map.png"))

    multi_mask = (multi_labels[:, :n_clusters] > 0.3).sum(axis=1) >= 2
    multi_count = np.sum(multi_mask)
    print(f"Multi-modal samples (≥2 clusters with >0.3 prob): {multi_count} / {len(probs)} ({100 * multi_count / len(probs):.2f}%)")
    np.save(os.path.join(output_dir, "multi_modal_mask.npy"), multi_mask)

if __name__ == "__main__":
    run_tactilecnn_with_normal(
        data_root="./organized_data/env_0",
        output_dir="./cluster/tactilecnn_output",
        n_clusters=5,
        auto_select_n=True
    )