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
def plot_activation_heatmap(weights_sequence, title, save_path, figsize=(12, 6)):
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

def save_activation_sequences(activation_records, output_dir):
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
            # 将样本按时间戳排序
            samples.sort(key=lambda x: (x['timestep'], x['sensor_id']))
            
            # 提取权重序列
            weights_sequence = np.stack([s['weights'] for s in samples])
            
            # 保存数据
            save_path = os.path.join(env_dir, f"{episode_name}_activations.npy")
            np.save(save_path, weights_sequence)
            
            # 生成可视化
            title = f'Prototype Activations - {env_name}/{episode_name}'
            plot_activation_heatmap(
                weights_sequence,
                title,
                save_path.replace('.npy', '.png')
            )