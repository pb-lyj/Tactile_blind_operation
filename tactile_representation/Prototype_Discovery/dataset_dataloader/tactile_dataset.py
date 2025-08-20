"""
触觉原型发现数据集
适配现有的力数据格式 (3, 20, 20)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob


class TactileForcesDataset(Dataset):
    """
    触觉力数据集，用于原型发现
    """
    
    def __init__(self, data_root, categories=None, start_frame=0, train_ratio=0.8, random_seed=42, is_train=True, augment=False, normalize_method='quantile_zscore'):
        """
        Args:
            data_root: 数据根目录路径 (data25.7_aligned)
            categories: 要包含的类别列表，如 ["cir_lar", "rect_med"] 等
            start_frame: 从第几帧开始截取数据
            train_ratio: 训练集比例，范围[0, 1]，用于控制数据划分
            random_seed: 随机种子，用于控制训练集/测试集的划分
            is_train: 是否加载训练集数据，True=训练集，False=测试集
            augment: 是否应用数据增强
            normalize_method: 归一化方法 ['quantile_zscore', 'minmax_255', 'channel_wise']
                - 'quantile_zscore': 分位数截断 + Z-score标准化（默认）
                - 'minmax_255': 0-255范围归一化，然后除以255（类似图像处理）
                - 'channel_wise': 按通道分别进行Min-Max归一化
        """
        self.samples = []
        self.augment = augment
        self.normalize_method = normalize_method
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.is_train = is_train
        
        # 设置随机种子以确保可重现的划分
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        if categories is None:
            categories = [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ]
        
        total_frames = 0
        valid_frames = 0
        total_episodes = 0
        train_episodes = 0
        test_episodes = 0
        
        for category in categories:
            category_path = os.path.join(data_root, category)
            if not os.path.exists(category_path):
                print(f"⚠️  警告: 类别目录不存在: {category_path}")
                continue
                
            # 获取该类别下的所有数据目录
            data_dirs = sorted([d for d in os.listdir(category_path) 
                              if os.path.isdir(os.path.join(category_path, d))])
            total_episodes += len(data_dirs)
            
            # 计算该类别的训练集数量
            category_train_count = int(len(data_dirs) * train_ratio)
            
            # 随机打乱数据目录顺序
            import random
            random.shuffle(data_dirs)
            
            # 划分训练集和测试集
            train_dirs = data_dirs[:category_train_count]
            test_dirs = data_dirs[category_train_count:]
            
            # 根据is_train参数选择使用训练集还是测试集
            selected_dirs = train_dirs if is_train else test_dirs
            
            for data_dir in selected_dirs:
                data_dir_path = os.path.join(category_path, data_dir)
                
                # 查找力数据文件 - 可能是forces.npy或_forces_l.npy/_forces_r.npy
                tactile_files = []
                forces_file = os.path.join(data_dir_path, "forces.npy")
                left_forces_file = os.path.join(data_dir_path, "_forces_l.npy")
                right_forces_file = os.path.join(data_dir_path, "_forces_r.npy")
                
                if os.path.exists(forces_file):
                    tactile_files.append(forces_file)
                elif os.path.exists(left_forces_file) and os.path.exists(right_forces_file):
                    tactile_files.extend([left_forces_file, right_forces_file])
                else:
                    continue
                    
                for tactile_file in tactile_files:
                    try:
                        data = np.load(tactile_file)  # shape (T, C, H, W) or (T, 6, 20, 20)
                        
                        # 处理不同的数据格式
                        if data.shape[1] == 6:  # 合并的左右手数据 (T, 6, 20, 20)
                            total_frames += data.shape[0]
                            
                            # 只使用从 start_frame 开始的数据
                            if data.shape[0] > start_frame:
                                for t in range(start_frame, data.shape[0]):
                                    frame = data[t]  # (6, 20, 20)
                                    
                                    # 分别提取左右手传感器的力数据
                                    left_sensor = frame[0:3]   # 左手传感器: channels 0,1,2
                                    right_sensor = frame[3:6]  # 右手传感器: channels 3,4,5
                                    
                                    self.samples.append(left_sensor)
                                    self.samples.append(right_sensor)
                                    valid_frames += 1
                        
                        elif data.shape[1] == 3:  # 单手数据 (T, 3, 20, 20)
                            total_frames += data.shape[0]
                            
                            # 只使用从 start_frame 开始的数据
                            if data.shape[0] > start_frame:
                                for t in range(start_frame, data.shape[0]):
                                    frame = data[t]  # (3, 20, 20)
                                    self.samples.append(frame)
                                    valid_frames += 1
                        
                    except Exception as e:
                        print(f"⚠️  警告: 无法加载 {tactile_file}: {e}")
                    continue
            
            train_episodes += len(train_dirs)
            test_episodes += len(test_dirs)
                    
        if len(self.samples) == 0:
            raise ValueError(f"未找到有效数据! 检查数据路径: {data_root}")
            
        self.samples = np.stack(self.samples)
        
        # 数据归一化 - 根据选择的方法进行调整
        self.samples = self._normalize_data(self.samples)
        
        print(f"[TactileForcesDataset] 数据统计:")
        print(f"  - 数据根目录: {data_root}")
        print(f"  - 包含类别: {categories}")
        print(f"  - 归一化方法: {self.normalize_method}")
        print(f"  - 总情节数: {total_episodes}")
        print(f"  - 训练集比例: {train_ratio:.1%}")
        print(f"  - 随机种子: {random_seed}")
        print(f"  - 训练集情节数: {train_episodes}")
        print(f"  - 测试集情节数: {test_episodes}")
        print(f"  - 当前加载: {'训练集' if is_train else '测试集'}")
        print(f"  - 总帧数: {total_frames}")
        print(f"  - 截取起始帧: {start_frame}")
        print(f"  - 有效帧数: {valid_frames}")
        print(f"  - 总样本数: {len(self.samples)} (包含左右手传感器)")
        print(f"  - 样本形状: {self.samples.shape}")
        print(f"  - 数据范围: [{self.samples.min():.4f}, {self.samples.max():.4f}]")

    def _normalize_data(self, data):
        """
        多种数据归一化处理方法
        Args:
            data: 原始数据 (N, 3, 20, 20)
        Returns:
            标准化后的数据
        """
        print(f"原始数据范围: [{data.min():.4f}, {data.max():.4f}]")
        print(f"使用归一化方法: {self.normalize_method}")
        
        if self.normalize_method == 'quantile_zscore':
            # 方法1: 分位数归一化 + Z-score标准化（原有方法）
            normalized_data = np.zeros_like(data)
            
            for i in range(data.shape[1]):  # 对每个通道分别处理
                channel_data = data[:, i, :, :]
                
                # 分位数截断 (去除极值)
                q1, q99 = np.percentile(channel_data, [1, 99])
                channel_data = np.clip(channel_data, q1, q99)
                
                # Z-score标准化
                mean = np.mean(channel_data)
                std = np.std(channel_data)
                if std > 1e-8:  # 避免除零
                    normalized_data[:, i, :, :] = (channel_data - mean) / std
                else:
                    normalized_data[:, i, :, :] = channel_data - mean
        
        elif self.normalize_method == 'minmax_255':
            # 方法2: 0-255范围归一化，然后除以255（类似图像处理）
            # 先映射到0-255范围
            data_min = data.min()
            data_max = data.max()
            if data_max - data_min > 1e-8:
                scaled_data = (data - data_min) / (data_max - data_min) * 255.0
            else:
                scaled_data = data - data_min
            
            # 然后除以255转换为float32范围[0,1]
            normalized_data = scaled_data.astype(np.float32) / 255.0
        
        elif self.normalize_method == 'channel_wise':
            # 方法3: 按通道分别进行Min-Max归一化
            normalized_data = np.zeros_like(data)
            
            for i in range(data.shape[1]):  # 对每个通道分别处理
                channel_data = data[:, i, :, :]
                
                # Min-Max归一化到[0,1]
                channel_min = channel_data.min()
                channel_max = channel_data.max()
                if channel_max - channel_min > 1e-8:
                    normalized_data[:, i, :, :] = (channel_data - channel_min) / (channel_max - channel_min)
                else:
                    normalized_data[:, i, :, :] = channel_data - channel_min
        
        else:
            raise ValueError(f"不支持的归一化方法: {self.normalize_method}")
                
        print(f"标准化后数据范围: [{normalized_data.min():.4f}, {normalized_data.max():.4f}]")
        
        # 检查是否有NaN或Inf
        if np.isnan(normalized_data).any() or np.isinf(normalized_data).any():
            print("⚠️  警告: 标准化后数据包含NaN或Inf，使用简单归一化...")
            normalized_data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        return normalized_data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]  # shape: (3, 20, 20)
        
        # 数据增强
        if self.augment and np.random.rand() > 0.5:
            sample = self._apply_augmentation(sample)
        
        # 转换为torch张量
        sample = torch.FloatTensor(sample)
        
        # 重建任务，输入和输出相同
        return {
            'image': sample,
            'target': sample  # 重建目标
        }

    def _apply_augmentation(self, sample):
        """
        应用数据增强
        """
        # 添加高斯噪声
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, sample.shape)
            sample = sample + noise
        
        # 随机翻转
        if np.random.rand() > 0.5:
            sample = np.flip(sample, axis=1)  # 水平翻转
        
        if np.random.rand() > 0.5:
            sample = np.flip(sample, axis=2)  # 垂直翻转
        
        return sample.copy()  # 确保返回连续内存


def create_train_test_tactile_datasets(data_root, categories=None, train_ratio=0.8, random_seed=42, 
                                     start_frame=0, augment_train=True, augment_test=False, 
                                     normalize_method='quantile_zscore'):
    """
    便捷函数：创建训练集和测试集的触觉数据集
    
    Args:
        data_root: 数据根目录路径
        categories: 要包含的类别列表
        train_ratio: 训练集比例
        random_seed: 随机种子
        start_frame: 起始帧
        augment_train: 训练集是否使用数据增强
        augment_test: 测试集是否使用数据增强
        normalize_method: 归一化方法
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_dataset = TactileForcesDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        train_ratio=train_ratio,
        random_seed=random_seed,
        is_train=True,
        augment=augment_train,
        normalize_method=normalize_method
    )
    
    test_dataset = TactileForcesDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        train_ratio=train_ratio,
        random_seed=random_seed,
        is_train=False,
        augment=augment_test,
        normalize_method=normalize_method
    )
    
    return train_dataset, test_dataset


class TactileSampleDataset(Dataset):
    """
    原始的触觉样本数据集 (向后兼容)
    """
    def __init__(self, root_dirs, start_frame=20):
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
