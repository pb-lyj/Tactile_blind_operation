import os
import sys
import time
import atexit
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import json

# =================== Config Json ====================
# 用于将 配置字典 保存为 JSON 文件
def save_config_to_json(config: dict, save_path: str, overwrite: bool = False):
    """
    将配置字典保存为 JSON 文件。
    """
    if os.path.exists(save_path) and not overwrite:
        raise FileExistsError(f"{save_path} already exists. Use overwrite=True to overwrite.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Config saved to {save_path}")
# =================== Logger ====================
# 用于将 print 输出同时写入 终端和文件
# try to do : 
#   sys.stdout = Logger("name.log")
class Logger:
    def __init__(self, filename="train.log", to_terminal=True, with_timestamp=True):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.to_terminal = to_terminal
        self.with_timestamp = with_timestamp
        atexit.register(self.close)

    def write(self, message):
        if self.with_timestamp and message.strip() != "":
            timestamp = time.strftime("[%Y-%m-%d %H:%M:%S] ")
            message = ''.join(timestamp + line if line.strip() else line for line in message.splitlines(True))

        if self.to_terminal:
            self.terminal.write(message)
        if not self.log.closed:
            self.log.write(message)

    def flush(self):
        if self.to_terminal:
            self.terminal.flush()
        if not self.log.closed:
            self.log.flush()

    def close(self):
        if self.log and not self.log.closed:
            self.log.close()


# =================== Physical XYZ Images ====================
# now for data_driven prototype discovery.py
def save_physicalXYZ_images(protos, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, proto in enumerate(protos):
        fx, fy, fz = proto
        fx *= 0.2
        fy *= 0.2
        skip = 2
        X, Y = np.meshgrid(np.arange(0, 32, skip), np.arange(0, 32, skip))
        U = fx[::skip, ::skip]
        V = fy[::skip, ::skip]
        C = np.abs(fz[::skip, ::skip])

        plt.figure(figsize=(3, 3))
        plt.quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy', scale=1, width=0.002, headwidth=3)
        plt.imshow(C, cmap='Reds', alpha=0.6)
        plt.title(f"Prototype {i}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"prototype_{i:02d}.png"))
        plt.close()

# =================== 原型激活序列保存和可视化 ====================
# now for validate_prototype.py
def plot_activation_heatmap(weights_sequence, title, save_path, figsize=(24, 12)):
    """绘制原型激活的热力图
    Args:
        weights_sequence: shape (T, num_prototypes) 的权重序列
        title: 图像标题
        save_path: 保存路径
        figsize: 图像尺寸, 默认 (12, 6)
    """
    plt.figure(figsize=figsize)
    plt.imshow(weights_sequence.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Activation Strength')
    plt.xlabel('Time Step')
    plt.ylabel('Prototype Index')
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def plot_dual_activation_heatmap(weights_sequence_left, weights_sequence_right, title, save_path, figsize=(24, 16)):
    """绘制左右传感器激活序列的对比图
    Args:
        weights_sequence_left: 左传感器激活序列, shape (T, num_prototypes)
        weights_sequence_right: 右传感器激活序列, shape (T, num_prototypes)
        title: 图像标题
        save_path: 保存路径
        figsize: 图像尺寸
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # 左传感器激活序列
    im1 = ax1.imshow(weights_sequence_left.T, aspect='auto', cmap='viridis')
    ax1.set_title('Left Sensor Activations')
    ax1.set_ylabel('Prototype Index')
    fig.colorbar(im1, ax=ax1, label='Activation Strength')
    
    # 右传感器激活序列
    im2 = ax2.imshow(weights_sequence_right.T, aspect='auto', cmap='viridis')
    ax2.set_title('Right Sensor Activations')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Prototype Index')
    fig.colorbar(im2, ax=ax2, label='Activation Strength')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_plot_activation_sequences(activation_records, output_dir):
    """保存和可视化原型激活序列
    Args:
        activation_records: 字典 {env_name: {episode_name: list of samples}}
        output_dir: 输出根目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for env_name, episodes in activation_records.items():
        env_dir = os.path.join(output_dir, env_name)
        os.makedirs(env_dir, exist_ok=True)
        
        for episode_name, samples in episodes.items():
            # 按时间戳和传感器ID排序
            samples.sort(key=lambda x: (x['timestep'], x['sensor_id']))
            
            # 分离左右传感器数据
            left_samples = [s['weights'] for s in samples if s['sensor_id'] == 0]
            right_samples = [s['weights'] for s in samples if s['sensor_id'] == 1]
            
            if len(left_samples) == len(right_samples):  # 确保数据对齐
                weights_sequence_left = np.stack(left_samples)
                weights_sequence_right = np.stack(right_samples)
                
                # 保存原始数据
                episode_dir = os.path.join(env_dir, episode_name)
                os.makedirs(episode_dir, exist_ok=True)
                np.save(os.path.join(episode_dir, "activation_sequence_left.npy"), weights_sequence_left)
                np.save(os.path.join(episode_dir, "activation_sequence_right.npy"), weights_sequence_right)
                
                # 生成对比可视化
                plot_dual_activation_heatmap(
                    weights_sequence_left,
                    weights_sequence_right,
                    f'Prototype Activations - {env_name}/{episode_name}',
                    os.path.join(episode_dir, "activation_sequence_comparison.png")
                )
            else:
                print(f"Warning: Unmatched sensor data in {env_name}/{episode_name}")

