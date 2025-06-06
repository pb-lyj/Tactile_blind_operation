import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
from glob import glob
# We use flatten and KMeans to cluster the tactile data

# ✅ 手动设定参数（修改此处即可）
NPY_ROOT_DIR = "./organized_data/env_0"
OUTPUT_DIR = "./cluster/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
N_CLUSTERS = 6

def load_all_npy(npy_dir):
    from glob import glob
    import numpy as np
    import os

    file_paths = sorted(glob(os.path.join(npy_dir, "episode_*", "tactile.npy")))
    data_list = []

    for path in file_paths:
        sample = np.load(path)  # shape: (T, 4, 6, 32, 32)
        if sample.ndim == 5:
            for t in range(sample.shape[0]):  # 遍历每个时间帧
                frame = sample[t]           # shape: (4, 6, 32, 32)
                last_patch = frame[-1]      # 只取 patch=3，即最后一帧 → (6, 32, 32)
                img1 = last_patch[0:3]      # (3, 32, 32)
                img2 = last_patch[3:6]      # (3, 32, 32)
                data_list.append(img1.flatten())  # → (3072,)
                data_list.append(img2.flatten())
        else:
            print(f"[WARN] Unexpected shape in {path}: {sample.shape}")

    if not data_list:
        raise ValueError("No valid data found. Check file shapes.")

    return np.stack(data_list), file_paths




def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X, file_paths = load_all_npy(NPY_ROOT_DIR)

    # PCA
    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(X)

    # KMeans
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=0)
    labels = kmeans.fit_predict(X_pca)

    # 保存标签
    label_path = os.path.join(OUTPUT_DIR, "cluster_labels.npy")
    np.save(label_path, labels)
    print(f"[✔] Cluster labels saved: {label_path}")

    # 评估
    sil_score = silhouette_score(X_pca, labels)
    print(f"[✔] Silhouette Score: {sil_score:.4f}")

    # t-SNE 可视化
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(X_pca)

    # 可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.colorbar(scatter)
    plt.title("Tactile Cluster Visualization (t-SNE)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    tsne_plot_path = os.path.join(OUTPUT_DIR, "tactile_clusters.png")
    plt.savefig(tsne_plot_path)
    print(f"[✔] t-SNE plot saved: {tsne_plot_path}")

if __name__ == "__main__":
    main()
