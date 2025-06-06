import os
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
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
        file_paths = sorted(glob(os.path.join(root_dir, "episode_*", "tactile.npy")))
        for path in file_paths:
            data = np.load(path)
            if data.ndim == 5:
                for t in range(data.shape[0]):
                    frame = data[t, -1]
                    self.samples.append(frame[0:3])
                    self.samples.append(frame[3:6])
        self.samples = np.stack(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.float32)

# === ResNet Encoder ===
class ResNetEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.base = resnet18(weights=None)
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base.maxpool = nn.Identity()
        self.base.fc = nn.Linear(self.base.fc.in_features, output_dim)

    def forward(self, x):
        return self.base(x)

# === Main ===
def run_resnet_gmm_clustering(data_root="./env_0", output_dir="./cluster/new_output", n_clusters=5):
    os.makedirs(output_dir, exist_ok=True)

    dataset = TactileDataset(data_root)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    model = ResNetEncoder(output_dim=128).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            feats = model(batch)
            all_features.append(feats.cpu().numpy())
    features = np.concatenate(all_features, axis=0)

    # Normalize + PCA
    features = StandardScaler().fit_transform(features)
    features_pca = PCA(n_components=50).fit_transform(features)

    # GMM Clustering
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    labels = gmm.fit_predict(features_pca)
    probs = gmm.predict_proba(features_pca)
    np.save(os.path.join(output_dir, "gmm_labels.npy"), labels)
    np.save(os.path.join(output_dir, "gmm_probs.npy"), probs)

    # Evaluation
    sil_score = silhouette_score(features_pca, labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    # Identify the "grasp" class: the class that appears in most samples
    dominant_cluster = np.argmax((probs > 0.3).sum(axis=0))
    print(f"Identified dominant 'grasp' cluster: {dominant_cluster}")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(features_pca)

    # Plot GMM Cluster Map
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter)
    plt.title("ResNet18 GMM Clusters (t-SNE)")
    plt.savefig(os.path.join(output_dir, "tactile_resnet18_gmm_tsne.png"))

    # Plot Grasp Class Soft Probability Heatmap
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=probs[:, dominant_cluster], cmap='Blues', alpha=0.8)
    plt.colorbar()
    plt.title(f"Grasp Cluster {dominant_cluster} - Soft Probability Map")
    plt.savefig(os.path.join(output_dir, "grasp_softmap.png"))

    # Compute entropy for each sample (uncertainty)
    ent = entropy(probs.T)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=ent, cmap='plasma', s=6)
    plt.colorbar()
    plt.title("Uncertainty (Entropy) Map")
    plt.savefig(os.path.join(output_dir, "uncertainty_entropy_map.png"))

    # Multi-label samples: those with >=2 classes above 0.3
    multi_mask = (probs > 0.3).sum(axis=1) >= 2
    multi_count = np.sum(multi_mask)
    print(f"Multi-modal samples (≥2 clusters with >0.3 prob): {multi_count} / {len(probs)} ({100 * multi_count / len(probs):.2f}%)")
    np.save(os.path.join(output_dir, "multi_modal_mask.npy"), multi_mask)

if __name__ == "__main__":
    run_resnet_gmm_clustering(data_root="./organized_data/env_0", output_dir="./cluster/new_output", n_clusters=5)
