import os
import sys
from datetime import datetime
# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
from Physical_mapping import Logger
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from data_driven_prototype_discovery import TactileSampleDataset
from Physical_mapping import save_plot_activation_sequences
from models.prototype_ae import PrototypeAutoencoder, compute_losses

# ==================== Config ====================
TEST_DATA_ROOT = "./organized_data_1_1"
MODEL_DIR = "./cluster/prototype_library/2025-06-21_22-35-18"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = os.path.join("./cluster/validation", timestamp)
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
                        self.sample_index.append((path, t, 0))  # sensor_left
                        self.sample_index.append((path, t, 1))  # sensor2_right
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
                sample = frame[0:3]  # sensor_left
            else:
                sample = frame[3:6]  # sensor_right
            sample[0:2] /= 0.1  # normalize XY
            return (
                torch.tensor(sample, dtype=torch.float32),
                (path, torch.tensor(t), torch.tensor(sensor_id))
            )
        else:
            return torch.tensor(self.samples[idx], dtype=torch.float32)

    
    
def load_compatible_model(model_dir):
    """加载兼容的模型，自动适配不同的模型结构"""
    
    # 检查模型目录中的文件
    prototype_path = os.path.join(model_dir, "prototypes.npy")
    model_path = os.path.join(model_dir, "prototype_model.pt")
    
    if not os.path.exists(prototype_path) or not os.path.exists(model_path):
        raise FileNotFoundError("找不到模型文件")
    
    # 从原型文件推断模型配置
    prototypes = np.load(prototype_path)
    num_prototypes = prototypes.shape[0]
    input_shape = prototypes.shape[1:]
    
    print(f"推断的模型配置:")
    print(f"  - 原型数量: {num_prototypes}")
    print(f"  - 输入形状: {input_shape}")
    
    # 加载权重检查模型结构
    checkpoint = torch.load(model_path)
    
    # 检查是否使用共享STN
    has_shared_stn = any(key.startswith('shared_stn') for key in checkpoint.keys())
    has_individual_stn = any(key.startswith('stn_modules') for key in checkpoint.keys())
    
    if has_shared_stn:
        share_stn = True
        print("检测到共享STN结构")
    elif has_individual_stn:
        share_stn = False
        print("检测到独立STN结构")
    else:
        share_stn = True  # 默认值
        print("无法确定STN结构，使用默认配置")
    
    # 创建匹配的模型
    model = PrototypeAutoencoder(
        num_prototypes=num_prototypes,
        input_shape=input_shape,
        share_stn=share_stn
    ).cuda()
    
    # 加载权重
    try:
        model.load_state_dict(checkpoint, strict=True)
        print("模型权重加载成功")
    except RuntimeError as e:
        print(f"严格加载失败，尝试宽松加载: {e}")
        model.load_state_dict(checkpoint, strict=False)
        print("模型权重宽松加载成功")
    
    return model, num_prototypes

def validate_model():
    # 加载测试数据
    test_dataset = MultiEnvTactileDataset(root_dir=TEST_DATA_ROOT, 
                                          exclude_envs=["env_0", "env_31", "env_32", "env_33", "env_34", "env_35", "env_36", "env_37", "env_38", "env_39", "env_40", "env_41", "env_42", "env_43", "env_44", "env_45", "env_46", "env_47", "env_48", "env_49"], 
                                          only_path=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # 数据统计初始化
    data_stats = {
        'total_environments': 0,
        'total_episodes': 0,
        'total_frames': 0,
        'valid_frames': 0,
        'environments': {}
    }

    # 统计原始数据
    env_dirs = sorted(glob(os.path.join(TEST_DATA_ROOT, "env_*")))
    data_stats['total_environments'] = len(env_dirs)
    
    for env in env_dirs:
        env_name = os.path.basename(env)
        if env_name == "env_0":  # 跳过训练环境
            continue
            
        data_stats['environments'][env_name] = {
            'episodes': 0,
            'frames': 0
        }
        
        episode_dirs = sorted(glob(os.path.join(env, "episode_*")))
        data_stats['environments'][env_name]['episodes'] = len(episode_dirs)
        data_stats['total_episodes'] += len(episode_dirs)
        
        for ep in episode_dirs:
            tactile_path = os.path.join(ep, "tactile.npy")
            if os.path.exists(tactile_path):
                try:
                    data = np.load(tactile_path)
                    frames = data.shape[0]
                    data_stats['environments'][env_name]['frames'] += frames
                    data_stats['total_frames'] += frames
                except Exception as e:
                    print(f"Warning: Could not load {tactile_path}: {e}")
    
    # 加载兼容模型
    model, actual_num_prototypes = load_compatible_model(MODEL_DIR)
    
    # 更新全局配置
    global NUM_PROTOTYPES
    NUM_PROTOTYPES = actual_num_prototypes
    
    model.eval()
    
    # 初始化损失统计和激活记录
    metrics_sum = {}
    total_samples = 0
    
    # 用于跟踪当前episode的数据
    current_env = None
    current_episode = None
    current_env_dir = None
    episode_samples = []
    
    with torch.no_grad():
        for batch_data in tqdm(test_loader, desc="Validating"):
            data, (paths, times, sensor_ids) = batch_data
            data = data.cuda()
            
            # 前向传播和损失计算
            transformed, weights, transformed_protos, thetas = model(data)
            _, batch_metrics = compute_losses(
                data, transformed, weights, 
                transformed_protos, thetas,
                diversity_lambda=0.1,
                entropy_lambda=0.05,
                stn_reg_lambda=0.02
            )
            
            # 累积指标
            batch_size = data.size(0)
            total_samples += batch_size
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0) + v * batch_size
            
            # 记录每个样本的激活情况
            for i in range(len(paths)):
                try:
                    path = paths[i]
                    t = times[i].item()
                    sensor_id = sensor_ids[i].item()
                    
                    env_name = path.split('/')[-3]
                    episode_name = path.split('/')[-2]
                    
                    # 如果环境发生变化，创建新的环境目录
                    if env_name != current_env:
                        current_env = env_name
                        current_env_dir = os.path.join(OUTPUT_DIR, "activations", env_name)
                        os.makedirs(current_env_dir, exist_ok=True)
                    
                    # 如果episode发生变化，保存前一个episode的数据并开始新的记录
                    if episode_name != current_episode:
                        if episode_samples:
                            # 保存前一个episode的数据
                            save_plot_activation_sequences(
                                {current_env: {current_episode: episode_samples}},
                                os.path.join(OUTPUT_DIR, "activations")
                            )
                        current_episode = episode_name
                        episode_samples = []
                    
                    # 收集当前样本数据
                    sample_info = {
                        'timestep': t,
                        'sensor_id': sensor_id,
                        'weights': weights[i].cpu().numpy()
                    }
                    episode_samples.append(sample_info)
                    
                except Exception as e:
                    print(f"Error processing path info at index {i}: {e}")
                    print(f"paths[{i}] = {paths[i]}")
                    continue
    
        # 保存最后一个episode的数据
        if episode_samples:
            save_plot_activation_sequences(
                {current_env: {current_episode: episode_samples}},
                os.path.join(OUTPUT_DIR, "activations")
            )
    
    # 记录有效处理的帧数
    data_stats['valid_frames'] = total_samples
    
    # 计算平均指标
    avg_metrics = {k: v/total_samples for k, v in metrics_sum.items()}
    
    # 打印结果
    results_str = "\n===== Validation Results =====\n"
    for k, v in avg_metrics.items():
        results_str += f"{k:20s}: {v:.4f}\n"
    
    print(results_str)
    
    # 保存结果
    os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)
    
    # 保存为txt文件
    metrics_path = os.path.join(OUTPUT_DIR, "metrics", "validation_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(results_str)
    
    # 保存为numpy格式
    np.save(os.path.join(OUTPUT_DIR, "metrics", "validation_metrics.npy"), avg_metrics)

    # 保存激活记录
    save_plot_activation_sequences(
        activation_records,
        os.path.join(OUTPUT_DIR, "activations")
    )
    
    # 打印数据统计
    stats_str = "\n===== Data Statistics =====\n"
    stats_str += f"Total Environments: {data_stats['total_environments']}\n"
    stats_str += f"Total Episodes: {data_stats['total_episodes']}\n"
    stats_str += f"Total Frames: {data_stats['total_frames']} (x2 sensors)\n"
    stats_str += f"Valid Processed Frames: {data_stats['valid_frames']}\n"
    stats_str += "\nPer Environment Statistics:\n"
    
    for env_name, env_stats in data_stats['environments'].items():
        stats_str += f"\n{env_name}:\n"
        stats_str += f"  Episodes: {env_stats['episodes']}\n"
        stats_str += f"  Frames: {env_stats['frames']} (x2 sensors)\n"
    
    print(stats_str)
    
    # 保存统计信息
    stats_path = os.path.join(OUTPUT_DIR, "metrics", "data_statistics.txt")
    with open(stats_path, "w") as f:
        f.write(stats_str)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sys.stdout = Logger(f"{OUTPUT_DIR}/validate_ae.log")
        # ======================= Report =======================
    """打印当前配置和输出目录"""
    print("=" * 50)
    print("Configuration:")
    print(f"TEST_DATA_ROOT   = {TEST_DATA_ROOT}")
    print(f"MODEL_DIR        = {MODEL_DIR}")
    print(f"OUTPUT_DIR       = {OUTPUT_DIR}")
    print(f"NUM_PROTOTYPES   = {NUM_PROTOTYPES}")
    print(f"BATCH_SIZE       = {BATCH_SIZE}")
    print("=" * 50)
    validate_model()