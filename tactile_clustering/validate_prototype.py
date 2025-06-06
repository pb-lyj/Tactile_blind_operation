import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from data_driven_prototype_discovery import (
    TactileSampleDataset, 
    PrototypeAutoencoder,
    compute_transform_consistency_loss
)
from Physical_mapping import save_activation_sequences

# ==================== Config ====================
TEST_DATA_ROOT = "./organized_data"
MODEL_DIR = "./cluster/prototype_library"
NUM_PROTOTYPES = 64
BATCH_SIZE = 256

class MultiEnvTactileDataset(TactileSampleDataset):
    def __init__(self, root_dir, exclude_envs=None, only_path=False):
        self.only_path = only_path
        exclude_envs = set(exclude_envs or [])

        env_dirs = sorted(glob(os.path.join(root_dir, "env_*")))
        env_dirs = [env for env in env_dirs if os.path.basename(env) not in exclude_envs]

        paths = []
        for env in env_dirs:
            episode_dirs = sorted(glob(os.path.join(env, "episode_*")))

            for ep in episode_dirs:
                tactile_path = os.path.join(ep, "tactile.npy")
                if os.path.exists(tactile_path):
                    paths.append(tactile_path)

        paths = sorted(paths)

        if self.only_path:
            self.sample_index = []  # 每项是 (path, t, sensor_id)
            for path in paths:
                try:
                    data = np.load(path, mmap_mode='r')
                    T = data.shape[0]
                    for t in range(T):
                        self.sample_index.append((path, t, 0))  # sensor1
                        self.sample_index.append((path, t, 1))  # sensor2
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")

        else:
            # 原始加载方式
            self.samples = []
            for path in paths:
                data = np.load(path)
                for t in range(data.shape[0]):
                    frame = data[t, -1]
                    self.samples.append(frame[0:3])
                    self.samples.append(frame[3:6])

            if len(self.samples) == 0:
                raise RuntimeError("No valid samples found.")
            self.samples = np.stack(self.samples)
            self.samples[:, 0:2] /= 0.1  # normalize XY

    def __len__(self):
        return len(self.sample_index) if self.only_path else len(self.samples)

    def __getitem__(self, idx):
        if self.only_path:
            path, t, sensor_id = self.sample_index[idx]
            data = np.load(path)
            frame = data[t, -1]  # shape (6, H, W)
            if sensor_id == 0:
                sample = frame[0:3]  # sensor1
            else:
                sample = frame[3:6]  # sensor2
            sample[0:2] /= 0.1  # normalize XY
            return (
                torch.tensor(sample, dtype=torch.float32),
                (path, torch.tensor(t), torch.tensor(sensor_id))
            )
        else:
            return torch.tensor(self.samples[idx], dtype=torch.float32)

    
    
def validate_model():
    # 加载测试数据
    test_dataset = MultiEnvTactileDataset(root_dir=TEST_DATA_ROOT, exclude_envs=["env_0"], only_path=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # 加载模型
    model = PrototypeAutoencoder(NUM_PROTOTYPES).cuda()
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "prototype_model.pt")))
    model.eval()
    
    # 初始化损失统计
    total_recon_loss = 0
    total_diversity_loss = 0
    total_transform_loss = 0
    total_samples = 0
    
    # 初始化激活跟踪字典
    activation_records = {}  # {env_name: {episode_name: list of weights}}
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Validating"):
            # 解包数据和路径信息
            data, (paths, times, sensor_ids) = batch_data  # 修改这里的解包方式
            data = data.cuda()
            
            # 前向传播
            transformed, weights, proto_combined, theta = model(data)
            weights = weights.cpu().numpy()  # (B, num_prototypes)
            
            # 记录每个样本的激活情况
            for i in range(len(paths)):
                try:
                    path = paths[i]
                    t = times[i].item()  # 从张量获取数值
                    sensor_id = sensor_ids[i].item()  # 从张量获取数值
                    
                    env_name = path.split('/')[-3]  # 获取 env_* 名称
                    episode_name = path.split('/')[-2]  # 获取 episode_* 名称
                    
                    if env_name not in activation_records:
                        activation_records[env_name] = {}
                    if episode_name not in activation_records[env_name]:
                        activation_records[env_name][episode_name] = []
                        
                    sample_info = {
                        'timestep': t,
                        'sensor_id': sensor_id,
                        'weights': weights[i]
                    }
                    activation_records[env_name][episode_name].append(sample_info)
                except Exception as e:
                    print(f"Error processing path info at index {i}: {e}")
                    print(f"paths[{i}] = {paths[i]}")
                    continue
            
            # 1. 重构损失
            recon_loss = torch.nn.functional.mse_loss(transformed, data)
            
            # 2. 原型多样性损失
            norm_prototypes = torch.nn.functional.normalize(
                model.prototypes.view(NUM_PROTOTYPES, -1), dim=1
            )
            similarity = torch.matmul(norm_prototypes, norm_prototypes.T)
            diversity_loss = (similarity**2).mean() - (1.0 / NUM_PROTOTYPES)
            
            # 3. 变换一致性损失
            transform_loss = compute_transform_consistency_loss(model.prototypes)
            
            # 累积损失
            batch_size = data.size(0)
            total_recon_loss += recon_loss.item() * batch_size
            total_diversity_loss += diversity_loss.item() * batch_size
            total_transform_loss += transform_loss.item() * batch_size
            total_samples += batch_size
    
    # 计算平均损失
    avg_recon_loss = total_recon_loss / total_samples
    avg_diversity_loss = total_diversity_loss / total_samples
    avg_transform_loss = total_transform_loss / total_samples
    total_loss = avg_recon_loss + 0.05 * avg_diversity_loss + 0.1 * avg_transform_loss
    
    # 打印结果
    results_str = "\n=== Validation Results ===\n"
    results_str += f"Reconstruction Loss: {avg_recon_loss:.4f}\n"
    results_str += f"Diversity Loss: {avg_diversity_loss:.4f}\n"
    results_str += f"Transform Consistency Loss: {avg_transform_loss:.4f}\n"
    results_str += f"Total Loss: {total_loss:.4f}\n"
    
    print(results_str)  # 在控制台显示结果
    
    # 保存结果
    os.makedirs("./validation_results", exist_ok=True)
    
    # 保存为txt文件
    with open("./validation_results/validation_results.txt", "w") as f:
        f.write(results_str)
    
    # 同时保存为numpy格式便于后续分析
    results = {
        "recon_loss": avg_recon_loss,
        "diversity_loss": avg_diversity_loss,
        "transform_loss": avg_transform_loss,
        "total_loss": total_loss
    }
    np.save("./validation_results/validation_losses.npy", results)

    # 保存激活记录
    save_activation_sequences(
        activation_records,
        "./validation_results/activations"
    )
    
if __name__ == "__main__":
    validate_model()