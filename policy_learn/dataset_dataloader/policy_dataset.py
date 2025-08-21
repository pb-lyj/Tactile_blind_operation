"""
策略学习数据集
加载data25.7_aligned中的数据，包括action、end_states、forces和resultant_force/moment
"""
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from glob import glob


class PolicyDataset(Dataset):
    """
    策略学习数据集，用于加载机器人运动策略相关的数据
    """
    
    def __init__(self, data_root, categories=None, start_frame=0, train_ratio=0.8, random_seed=42, is_train=True, use_end_states=True, use_forces=True, use_resultants=True, 
                 normalize_config=None):
        """
        Args:
            data_root: 数据根目录路径 (data25.7_aligned)
            categories: 要包含的类别列表，如 ["cir_lar", "rect_med"] 等
            start_frame: 从第几帧开始截取数据
            train_ratio: 训练集比例，范围[0, 1]，用于控制数据划分
            random_seed: 随机种子，用于控制训练集/测试集的划分
            is_train: 是否加载训练集数据，True=训练集，False=测试集
            use_end_states: 是否加载机器人末端状态数据 (_end_states.npy)
            use_forces: 是否加载触觉力数据 (_forces_l.npy/_forces_r.npy)
            use_resultants: 是否加载合力/合力矩数据 (_resultant_force_*.npy/_resultant_moment_*.npy)
            normalize_config: 归一化配置字典，格式如下：
                {
                    'forces': 'zscore',      # forces数据的归一化方法
                    'actions': 'minmax',     # actions数据的归一化方法
                    'end_states': None,      # end_states数据的归一化方法（None表示不归一化）
                    'resultants': 'zscore'   # resultants数据的归一化方法
                }
                支持的归一化方法：'zscore', 'minmax', None
                - 'zscore': Z-score标准化 (mean=0, std=1)
                - 'minmax': MinMax归一化到[-1, 1]区间，保存min/max值用于重映射
                - None: 不进行归一化
        """
        self.data_samples = []  # 存储所有样本路径
        self.start_frame = start_frame
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.is_train = is_train
        self.use_end_states = use_end_states
        self.use_forces = use_forces
        self.use_resultants = use_resultants
        
        # 设置默认归一化配置
        if normalize_config is None:
            normalize_config = {
                'forces': None,      # 默认不归一化forces
                'actions': None,     # 默认不归一化actions 
                'end_states': None,  # 默认不归一化end_states
                'resultants': None   # 默认不归一化resultants
            }
        self.normalize_config = normalize_config
        
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
                
                # 确认所有必要的文件都存在
                action_file = os.path.join(data_dir_path, "_action.npy")
                required_files = [action_file]
                
                if use_end_states:
                    end_states_file = os.path.join(data_dir_path, "_end_states.npy")
                    required_files.append(end_states_file)
                
                missing_files = [f for f in required_files if not os.path.exists(f)]
                if missing_files:
                    print(f"⚠️  警告: 缺少必要的数据文件: {missing_files}")
                    continue
                    
                # 将有效的数据目录添加到样本列表
                self.data_samples.append({
                    'path': data_dir_path,
                    'category': category
                })
            
            train_episodes += len(train_dirs)
            test_episodes += len(test_dirs)
        
        print(f"[PolicyDataset] 数据统计:")
        print(f"  - 数据根目录: {data_root}")
        print(f"  - 包含类别: {categories}")
        print(f"  - 总情节数: {total_episodes}")
        print(f"  - 训练集比例: {train_ratio:.1%}")
        print(f"  - 随机种子: {random_seed}")
        print(f"  - 训练集情节数: {train_episodes}")
        print(f"  - 测试集情节数: {test_episodes}")
        print(f"  - 当前加载: {'训练集' if is_train else '测试集'}")
        print(f"  - 有效情节数: {len(self.data_samples)}")
        print(f"  - 截取起始帧: {start_frame}")
        print(f"  - 加载末端状态: {use_end_states}")
        print(f"  - 加载力数据: {use_forces}")
        print(f"  - 加载合力/矩数据: {use_resultants}")

    def __len__(self):
        return len(self.data_samples)

    def _normalize_data(self, data, data_type):
        """
        数据归一化处理方法
        Args:
            data: 原始数据（numpy数组）
            data_type: 数据类型，用于选择归一化方法 ('forces', 'actions', 'end_states', 'resultants')
        Returns:
            标准化后的数据
        """
        normalize_method = self.normalize_config.get(data_type, None)
        
        # 如果不需要归一化，直接返回原数据
        if normalize_method is None:
            return data
        
        print(f"对{data_type}数据应用{normalize_method}归一化...")
        print(f"原始数据范围: [{data.min():.6f}, {data.max():.6f}]")
        
        if normalize_method == 'zscore':
            # Z-score标准化: (x - mean) / std
            mean = np.mean(data)
            std = np.std(data)
            if std > 1e-8:  # 避免除零
                normalized_data = (data - mean) / std
            else:
                normalized_data = data - mean
                
            print(f"Z-score归一化: mean={mean:.6f}, std={std:.6f}")
            
        elif normalize_method == 'minmax':
            # MinMax归一化到[-1, 1]区间
            data_min = np.min(data)
            data_max = np.max(data)
            
            # 保存min/max值用于重映射
            if not hasattr(self, 'normalization_params'):
                self.normalization_params = {}
            self.normalization_params[data_type] = {
                'min': data_min,
                'max': data_max,
                'method': 'minmax'
            }
            
            if data_max - data_min > 1e-8:
                # 归一化到[-1, 1]: 2 * (x - min) / (max - min) - 1
                normalized_data = 2.0 * (data - data_min) / (data_max - data_min) - 1.0
            else:
                normalized_data = np.zeros_like(data)
                
            print(f"MinMax归一化到[-1,1]: min={data_min:.6f}, max={data_max:.6f}")
            
        else:
            raise ValueError(f"不支持的归一化方法: {normalize_method}. 支持的方法: 'zscore', 'minmax'")
                
        print(f"归一化后数据范围: [{normalized_data.min():.6f}, {normalized_data.max():.6f}]")
        
        # 检查是否有NaN或Inf
        if np.isnan(normalized_data).any() or np.isinf(normalized_data).any():
            print("⚠️  警告: 归一化后数据包含NaN或Inf，使用原始数据...")
            normalized_data = data
        
        return normalized_data
    
    def denormalize_data(self, normalized_data, data_type):
        """
        将归一化后的数据重映射回原始范围
        Args:
            normalized_data: 归一化后的数据
            data_type: 数据类型
        Returns:
            重映射后的原始范围数据
        """
        if not hasattr(self, 'normalization_params') or data_type not in self.normalization_params:
            # 如果没有归一化参数，直接返回
            return normalized_data
        
        params = self.normalization_params[data_type]
        
        if params['method'] == 'minmax':
            # 从[-1, 1]重映射回原始范围: (x + 1) / 2 * (max - min) + min
            data_min = params['min']
            data_max = params['max']
            original_data = (normalized_data + 1.0) / 2.0 * (data_max - data_min) + data_min
            return original_data
        elif params['method'] == 'zscore':
            # Z-score反归一化: x * std + mean
            mean = params['mean']
            std = params['std']
            original_data = normalized_data * std + mean
            return original_data
        else:
            return normalized_data

    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Returns:
            dict: 包含以下键：
                - 'action': (N, 6) 机器人动作数据（不含时间戳）
                - 'timestamps': (N+1,) 时间戳序列（如果use_end_states=True）或 (N,)（如果use_end_states=False）
                - 'category': 类别名称
                - 如果use_end_states=True:
                  - 'end_states': (N+1, 6) 机器人末端状态（不含时间戳）
                - 如果use_forces=True:
                  - 'forces_l': (N+1, 3, 20, 20) 左手触觉传感器数据（如果use_end_states=True）或 (N, 3, 20, 20)
                  - 'forces_r': (N+1, 3, 20, 20) 右手触觉传感器数据（如果use_end_states=True）或 (N, 3, 20, 20)
                - 如果use_resultants=True:
                  - 'resultant_force_l': (N+1, 3) 左手合力（如果use_end_states=True）或 (N, 3)
                  - 'resultant_force_r': (N+1, 3) 右手合力（如果use_end_states=True）或 (N, 3)
                  - 'resultant_moment_l': (N+1, 3) 左手合力矩（如果use_end_states=True）或 (N, 3)
                  - 'resultant_moment_r': (N+1, 3) 右手合力矩（如果use_end_states=True）或 (N, 3)
        """
        sample_info = self.data_samples[idx]
        dir_path = sample_info['path']
        
        # 加载动作数据
        action_data = np.load(os.path.join(dir_path, "_action.npy"))
        action_data = action_data[self.start_frame:]
        actions = action_data[:, 1:]  # 去掉时间戳列
        
        # 应用归一化
        actions = self._normalize_data(actions, 'actions')
        
        # 创建返回字典
        result = {
            'action': torch.FloatTensor(actions),
            'category': sample_info['category']
        }
        
        # 如果需要，加载末端状态数据
        if self.use_end_states:
            end_states_data = np.load(os.path.join(dir_path, "_end_states.npy"))
            end_states_data = end_states_data[self.start_frame:]
            timestamps = end_states_data[:, 0]  # 使用end_states的时间戳
            end_states = end_states_data[:, 1:]  # 去掉时间戳列
            
            # 应用归一化
            end_states = self._normalize_data(end_states, 'end_states')
            
            result['end_states'] = torch.FloatTensor(end_states)
            result['timestamps'] = torch.FloatTensor(timestamps)
        else:
            # 如果不加载end_states，使用action的时间戳
            action_timestamps = action_data[:, 0]
            result['timestamps'] = torch.FloatTensor(action_timestamps)
        
        # 如果需要，加载力数据
        if self.use_forces:
            if self.use_end_states:
                # 如果有end_states，使用end_states的长度
                forces_l = np.load(os.path.join(dir_path, "_forces_l.npy"))[self.start_frame:]
                forces_r = np.load(os.path.join(dir_path, "_forces_r.npy"))[self.start_frame:]
            else:
                # 如果没有end_states，使用action的长度
                forces_l = np.load(os.path.join(dir_path, "_forces_l.npy"))[self.start_frame:self.start_frame+len(actions)]
                forces_r = np.load(os.path.join(dir_path, "_forces_r.npy"))[self.start_frame:self.start_frame+len(actions)]
            
            # 应用归一化
            forces_l = self._normalize_data(forces_l, 'forces')
            forces_r = self._normalize_data(forces_r, 'forces')
            
            result['forces_l'] = torch.FloatTensor(forces_l)
            result['forces_r'] = torch.FloatTensor(forces_r)
        
        # 如果需要，加载合力和合力矩数据
        if self.use_resultants:
            if self.use_end_states:
                # 如果有end_states，使用end_states的长度
                resultant_force_l = np.load(os.path.join(dir_path, "_resultant_force_l.npy"))[self.start_frame:]
                resultant_force_r = np.load(os.path.join(dir_path, "_resultant_force_r.npy"))[self.start_frame:]
                resultant_moment_l = np.load(os.path.join(dir_path, "_resultant_moment_l.npy"))[self.start_frame:]
                resultant_moment_r = np.load(os.path.join(dir_path, "_resultant_moment_r.npy"))[self.start_frame:]
            else:
                # 如果没有end_states，使用action的长度
                resultant_force_l = np.load(os.path.join(dir_path, "_resultant_force_l.npy"))[self.start_frame:self.start_frame+len(actions)]
                resultant_force_r = np.load(os.path.join(dir_path, "_resultant_force_r.npy"))[self.start_frame:self.start_frame+len(actions)]
                resultant_moment_l = np.load(os.path.join(dir_path, "_resultant_moment_l.npy"))[self.start_frame:self.start_frame+len(actions)]
                resultant_moment_r = np.load(os.path.join(dir_path, "_resultant_moment_r.npy"))[self.start_frame:self.start_frame+len(actions)]
            
            # 应用归一化
            resultant_force_l = self._normalize_data(resultant_force_l, 'resultants')
            resultant_force_r = self._normalize_data(resultant_force_r, 'resultants')
            resultant_moment_l = self._normalize_data(resultant_moment_l, 'resultants')
            resultant_moment_r = self._normalize_data(resultant_moment_r, 'resultants')
            
            result['resultant_force_l'] = torch.FloatTensor(resultant_force_l)
            result['resultant_force_r'] = torch.FloatTensor(resultant_force_r)
            result['resultant_moment_l'] = torch.FloatTensor(resultant_moment_l)
            result['resultant_moment_r'] = torch.FloatTensor(resultant_moment_r)
            
        return result


class PolicyBatchedDataset(Dataset):
    """
    策略学习批量数据集，对单个数据序列进行分割为固定长度的序列片段
    支持两种时序模式：严格时序和随机采样
    """
    
    def __init__(self, data_root, categories=None, sequence_length=10, stride=5, 
                 start_frame=0, train_ratio=0.8, random_seed=42, is_train=True, use_end_states=True, use_forces=True, use_resultants=True,
                 chronology=True):
        """
        Args:
            data_root: 数据根目录路径 (data25.7_aligned)
            categories: 要包含的类别列表，如 ["cir_lar", "rect_med"] 等
            sequence_length: 每个序列片段的长度
            stride: 序列分割的步长
            start_frame: 从第几帧开始截取数据
            train_ratio: 训练集比例，范围[0, 1]，用于控制数据划分
            random_seed: 随机种子，用于控制训练集/测试集的划分
            is_train: 是否加载训练集数据，True=训练集，False=测试集
            use_end_states: 是否加载机器人末端状态数据 (_end_states.npy)
            use_forces: 是否加载触觉力数据 (_forces_l.npy/_forces_r.npy)
            use_resultants: 是否加载合力/合力矩数据 (_resultant_force_*.npy/_resultant_moment_*.npy)
            chronology: 时序模式控制
                - True: 严格时序模式，每个episode内的序列按顺序训练，不允许跨episode连接
                - False: 随机采样模式，允许打乱和连接，只要当前帧的输入和action对齐即可
        """
        self.base_dataset = PolicyDataset(
            data_root, categories, start_frame, train_ratio, random_seed, is_train,
            use_end_states, use_forces, use_resultants
        )
        self.sequence_length = sequence_length
        self.stride = stride
        self.chronology = chronology
        self.sequence_indices = []
        
        total_episodes = 0
        total_sequences = 0
        
        if chronology:
            # 严格时序模式：每个episode内按顺序分割，不跨越episode边界
            for sample_idx in range(len(self.base_dataset)):
                # 获取样本长度（不加载数据，仅检查文件大小）
                sample_info = self.base_dataset.data_samples[sample_idx]
                action_path = os.path.join(sample_info['path'], "_action.npy")
                episode_length = np.load(action_path).shape[0] - start_frame
                
                # 计算该episode内可以分割出的序列数量
                num_sequences = max(0, (episode_length - sequence_length) // stride + 1)
                episode_sequences = 0
                
                # 为该episode生成所有有效的序列片段
                for seq_start in range(0, num_sequences * stride, stride):
                    seq_end = seq_start + sequence_length
                    # 确保序列完全在episode边界内
                    if seq_end <= episode_length:
                        self.sequence_indices.append((sample_idx, seq_start, seq_end))
                        episode_sequences += 1
                
                if episode_sequences > 0:
                    total_episodes += 1
                    total_sequences += episode_sequences
        else:
            # 随机采样模式：将所有数据合并，允许跨episode连接
            all_frames = []  # 存储所有帧的信息 (sample_idx, frame_idx)
            
            for sample_idx in range(len(self.base_dataset)):
                sample_info = self.base_dataset.data_samples[sample_idx]
                action_path = os.path.join(sample_info['path'], "_action.npy")
                episode_length = np.load(action_path).shape[0] - start_frame
                
                # 将该episode的所有帧添加到全局帧列表
                for frame_idx in range(episode_length):
                    all_frames.append((sample_idx, frame_idx))
                
                total_episodes += 1
            
            # 从全局帧列表中按stride采样序列
            total_frames = len(all_frames)
            num_sequences = max(0, (total_frames - sequence_length) // stride + 1)
            
            for seq_start in range(0, num_sequences * stride, stride):
                seq_end = seq_start + sequence_length
                if seq_end <= total_frames:
                    # 存储序列的起始和结束在全局帧列表中的位置
                    self.sequence_indices.append(('global', seq_start, seq_end))
                    total_sequences += 1
            
            # 存储全局帧列表以备后用
            self.all_frames = all_frames
        
        print(f"[PolicyBatchedDataset] 数据统计:")
        print(f"  - 时序模式: {'严格时序' if chronology else '随机采样'}")
        print(f"  - 序列长度: {sequence_length}")
        print(f"  - 序列步长: {stride}")
        print(f"  - 有效episodes: {total_episodes}")
        print(f"  - 总序列数: {len(self.sequence_indices)}")
        if total_episodes > 0:
            print(f"  - 平均每episode序列数: {total_sequences / total_episodes:.1f}")
        
        # 统计每个类别的序列分布
        if chronology:
            category_stats = {}
            for sample_idx, _, _ in self.sequence_indices:
                category = self.base_dataset.data_samples[sample_idx]['category']
                category_stats[category] = category_stats.get(category, 0) + 1
        else:
            category_stats = {}
            for _, start_idx, _ in self.sequence_indices:
                # 对于随机采样模式，统计序列起始帧所属的类别
                sample_idx, _ = all_frames[start_idx]
                category = self.base_dataset.data_samples[sample_idx]['category']
                category_stats[category] = category_stats.get(category, 0) + 1
        
        print(f"  - 类别分布:")
        for category, count in sorted(category_stats.items()):
            print(f"    * {category}: {count} 序列")
    
    def __len__(self):
        return len(self.sequence_indices)
    
    def __getitem__(self, idx):
        """
        获取指定索引的序列片段
        
        Returns:
            dict: 包含以下键：
                - 'action': (sequence_length, 6) 机器人动作序列
                - 'timestamps': (sequence_length+1,) 或 (sequence_length,) 时间戳序列
                - 'category': 类别名称
                - 'episode_info': episode相关信息
                - 如果use_end_states=True:
                  - 'end_states': (sequence_length+1, 6) 机器人末端状态序列
                - 如果use_forces=True:
                  - 'forces_l': (sequence_length+1, 3, 20, 20) 或 (sequence_length, 3, 20, 20) 左手触觉传感器数据
                  - 'forces_r': (sequence_length+1, 3, 20, 20) 或 (sequence_length, 3, 20, 20) 右手触觉传感器数据
                - 如果use_resultants=True:
                  - 'resultant_force_l': (sequence_length+1, 3) 或 (sequence_length, 3) 左手合力
                  - 'resultant_force_r': (sequence_length+1, 3) 或 (sequence_length, 3) 右手合力
                  - 'resultant_moment_l': (sequence_length+1, 3) 或 (sequence_length, 3) 左手合力矩
                  - 'resultant_moment_r': (sequence_length+1, 3) 或 (sequence_length, 3) 右手合力矩
        """
        if self.chronology:
            # 严格时序模式：从单个episode中获取连续序列
            sample_idx, start, end = self.sequence_indices[idx]
            full_sample = self.base_dataset[sample_idx]
            
            # 对所有数据进行切片处理
            result = {
                'action': full_sample['action'][start:end],
                'timestamps': full_sample['timestamps'][start:end+1] if 'end_states' in full_sample else full_sample['timestamps'][start:end],
                'category': full_sample['category'],
                'episode_info': {
                    'episode_idx': sample_idx,
                    'seq_start': start,
                    'seq_end': end,
                    'is_continuous': True
                }
            }
            
            # 处理可选的end_states数据
            if 'end_states' in full_sample:
                result['end_states'] = full_sample['end_states'][start:end+1]  # 多取一个状态
            
            # 处理可选数据
            if 'forces_l' in full_sample:
                if 'end_states' in full_sample:
                    result['forces_l'] = full_sample['forces_l'][start:end+1]
                    result['forces_r'] = full_sample['forces_r'][start:end+1]
                else:
                    result['forces_l'] = full_sample['forces_l'][start:end]
                    result['forces_r'] = full_sample['forces_r'][start:end]
                
            if 'resultant_force_l' in full_sample:
                if 'end_states' in full_sample:
                    result['resultant_force_l'] = full_sample['resultant_force_l'][start:end+1]
                    result['resultant_force_r'] = full_sample['resultant_force_r'][start:end+1]
                    result['resultant_moment_l'] = full_sample['resultant_moment_l'][start:end+1]
                    result['resultant_moment_r'] = full_sample['resultant_moment_r'][start:end+1]
                else:
                    result['resultant_force_l'] = full_sample['resultant_force_l'][start:end]
                    result['resultant_force_r'] = full_sample['resultant_force_r'][start:end]
                    result['resultant_moment_l'] = full_sample['resultant_moment_l'][start:end]
                    result['resultant_moment_r'] = full_sample['resultant_moment_r'][start:end]
                
        else:
            # 随机采样模式：可能跨越多个episode
            _, global_start, global_end = self.sequence_indices[idx]
            
            # 收集序列中的所有帧
            if self.base_dataset.use_end_states:
                sequence_frames = self.all_frames[global_start:global_end+1]  # +1是为了包含多一个状态
            else:
                sequence_frames = self.all_frames[global_start:global_end]  # 不需要额外状态
            
            actions = []
            end_states = [] if self.base_dataset.use_end_states else None
            timestamps = []
            forces_l = [] if self.base_dataset.use_forces else None
            forces_r = [] if self.base_dataset.use_forces else None
            resultant_force_l = [] if self.base_dataset.use_resultants else None
            resultant_force_r = [] if self.base_dataset.use_resultants else None
            resultant_moment_l = [] if self.base_dataset.use_resultants else None
            resultant_moment_r = [] if self.base_dataset.use_resultants else None
            
            # 记录episode信息
            episode_changes = []
            prev_episode = None
            
            for i, (sample_idx, frame_idx) in enumerate(sequence_frames):
                # 检测episode切换
                if prev_episode is not None and prev_episode != sample_idx:
                    episode_changes.append(i)
                prev_episode = sample_idx
                
                # 获取该帧的数据
                full_sample = self.base_dataset[sample_idx]
                
                # 收集状态数据（仅当use_end_states=True时）
                if self.base_dataset.use_end_states and 'end_states' in full_sample and frame_idx < len(full_sample['end_states']):
                    end_states.append(full_sample['end_states'][frame_idx])
                
                # 收集时间戳数据
                if frame_idx < len(full_sample['timestamps']):
                    timestamps.append(full_sample['timestamps'][frame_idx])
                    
                    if forces_l is not None and 'forces_l' in full_sample and frame_idx < len(full_sample['forces_l']):
                        forces_l.append(full_sample['forces_l'][frame_idx])
                        forces_r.append(full_sample['forces_r'][frame_idx])
                    
                    if resultant_force_l is not None and 'resultant_force_l' in full_sample and frame_idx < len(full_sample['resultant_force_l']):
                        resultant_force_l.append(full_sample['resultant_force_l'][frame_idx])
                        resultant_force_r.append(full_sample['resultant_force_r'][frame_idx])
                        resultant_moment_l.append(full_sample['resultant_moment_l'][frame_idx])
                        resultant_moment_r.append(full_sample['resultant_moment_r'][frame_idx])
                
                # 收集动作数据
                collect_action = False
                if self.base_dataset.use_end_states:
                    # 如果使用end_states，动作比状态少一个
                    collect_action = i < len(sequence_frames) - 1 and frame_idx < len(full_sample['action'])
                else:
                    # 如果不使用end_states，动作和其他数据同样多
                    collect_action = frame_idx < len(full_sample['action'])
                
                if collect_action:
                    actions.append(full_sample['action'][frame_idx])
            
            # 创建结果字典
            result = {
                'action': torch.stack(actions) if actions else torch.empty(0, full_sample['action'].shape[-1]),
                'timestamps': torch.stack(timestamps) if timestamps else torch.empty(0),
                'category': 'mixed' if len(episode_changes) > 0 else full_sample['category'],
                'episode_info': {
                    'episode_changes': episode_changes,
                    'num_episodes': len(set(sample_idx for sample_idx, _ in sequence_frames)),
                    'is_continuous': len(episode_changes) == 0
                }
            }
            
            # 处理可选的end_states数据
            if end_states is not None and end_states:
                result['end_states'] = torch.stack(end_states)
            elif self.base_dataset.use_end_states:
                result['end_states'] = torch.empty(0, full_sample['end_states'].shape[-1] if 'end_states' in full_sample else 6)
            
            # 处理可选数据
            if forces_l is not None and forces_l:
                result['forces_l'] = torch.stack(forces_l)
                result['forces_r'] = torch.stack(forces_r)
                
            if resultant_force_l is not None and resultant_force_l:
                result['resultant_force_l'] = torch.stack(resultant_force_l)
                result['resultant_force_r'] = torch.stack(resultant_force_r)
                result['resultant_moment_l'] = torch.stack(resultant_moment_l)
                result['resultant_moment_r'] = torch.stack(resultant_moment_r)
        
        return result


def create_train_test_datasets(data_root, categories=None, train_ratio=0.8, random_seed=42,
                              start_frame=0, use_end_states=True, use_forces=True, use_resultants=True,
                              normalize_config=None):
    """
    便捷函数：创建训练集和测试集数据集
    
    Args:
        data_root: 数据根目录路径
        categories: 要包含的类别列表
        train_ratio: 训练集比例
        random_seed: 随机种子
        start_frame: 起始帧
        use_end_states: 是否加载末端状态
        use_forces: 是否加载力数据
        use_resultants: 是否加载合力/矩数据
        normalize_config: 归一化配置字典
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_dataset = PolicyDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        train_ratio=train_ratio,
        random_seed=random_seed,
        is_train=True,
        use_end_states=use_end_states,
        use_forces=use_forces,
        use_resultants=use_resultants,
        normalize_config=normalize_config
    )
    
    test_dataset = PolicyDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        train_ratio=train_ratio,
        random_seed=random_seed,
        is_train=False,
        use_end_states=use_end_states,
        use_forces=use_forces,
        use_resultants=use_resultants,
        normalize_config=normalize_config
    )
    
    return train_dataset, test_dataset


def create_train_test_batched_datasets(data_root, categories=None, sequence_length=10, stride=5,
                                     train_ratio=0.8, random_seed=42, start_frame=0, 
                                     use_end_states=True, use_forces=True, use_resultants=True,
                                     chronology=True):
    """
    便捷函数：创建训练集和测试集的批量数据集
    
    Args:
        data_root: 数据根目录路径
        categories: 要包含的类别列表
        sequence_length: 序列长度
        stride: 序列步长
        train_ratio: 训练集比例
        random_seed: 随机种子
        start_frame: 起始帧
        use_end_states: 是否加载末端状态
        use_forces: 是否加载力数据
        use_resultants: 是否加载合力/矩数据
        chronology: 时序模式
    
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    train_dataset = PolicyBatchedDataset(
        data_root=data_root,
        categories=categories,
        sequence_length=sequence_length,
        stride=stride,
        start_frame=start_frame,
        train_ratio=train_ratio,
        random_seed=random_seed,
        is_train=True,
        use_end_states=use_end_states,
        use_forces=use_forces,
        use_resultants=use_resultants,
        chronology=chronology
    )
    
    test_dataset = PolicyBatchedDataset(
        data_root=data_root,
        categories=categories,
        sequence_length=sequence_length,
        stride=stride,
        start_frame=start_frame,
        train_ratio=train_ratio,
        random_seed=random_seed,
        is_train=False,
        use_end_states=use_end_states,
        use_forces=use_forces,
        use_resultants=use_resultants,
        chronology=chronology
    )
    
    return train_dataset, test_dataset


# 使用示例
if __name__ == "__main__":
    import sys
    import os
    
    # 设置数据根目录
    data_root = "../../datasets/data25.7_aligned"
    
    # 创建数据集 - 包含所有数据（训练集）
    dataset_full = PolicyDataset(
        data_root=data_root,
        categories=["cir_lar", "cir_med"],
        start_frame=0,
        train_ratio=0.8,
        random_seed=42,
        is_train=True,
        use_end_states=True,
        use_forces=True,
        use_resultants=True
    )
    
    # 打印第一个样本的信息
    sample_full = dataset_full[0]
    print("\n完整样本信息:")
    for k, v in sample_full.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  - {k}: {v}")
    
    # 创建数据集 - 仅包含动作数据（测试集）
    print("\n=== 仅动作模式（测试集）===")
    dataset_action_only = PolicyDataset(
        data_root=data_root,
        categories=["cir_lar", "cir_med"],
        start_frame=0,
        train_ratio=0.8,
        random_seed=42,
        is_train=False,
        use_end_states=False,
        use_forces=False,
        use_resultants=False
    )
    
    # 打印第一个样本的信息
    sample_action = dataset_action_only[0]
    print("\n仅动作样本信息:")
    for k, v in sample_action.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"  - {k}: {v}")
    
    # 创建批量数据集 - 严格时序模式（包含完整数据）
    print("\n=== 严格时序模式（完整数据）===")
    batched_dataset_chrono = PolicyBatchedDataset(
        data_root=data_root,
        categories=["cir_lar", "cir_med"],
        sequence_length=10,
        stride=5,
        start_frame=0,
        train_ratio=0.8,
        random_seed=42,
        is_train=True,
        use_end_states=True,
        use_forces=True,
        use_resultants=True,
        chronology=True
    )
    
    # 打印第一个批次的信息
    batch_chrono = batched_dataset_chrono[0]
    print("\n时序批次信息:")
    for k, v in batch_chrono.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  - {k}: {v}")
        else:
            print(f"  - {k}: {v}")
    
    # 创建批量数据集 - 仅动作模式（测试集）
    print("\n=== 仅动作序列模式（测试集）===")
    batched_dataset_action = PolicyBatchedDataset(
        data_root=data_root,
        categories=["cir_lar", "cir_med"],
        sequence_length=10,
        stride=5,
        start_frame=0,
        train_ratio=0.8,
        random_seed=42,
        is_train=False,
        use_end_states=False,
        use_forces=False,
        use_resultants=False,
        chronology=True
    )
    
    # 打印第一个批次的信息
    batch_action = batched_dataset_action[0]
    print("\n仅动作批次信息:")
    for k, v in batch_action.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  - {k}: {v}")
        else:
            print(f"  - {k}: {v}")
    
    # 创建批量数据集 - 随机采样模式（完整数据）
    print("\n=== 随机采样模式（完整数据）===")
    batched_dataset_random = PolicyBatchedDataset(
        data_root=data_root,
        categories=["cir_lar", "cir_med"],
        sequence_length=10,
        stride=5,
        start_frame=0,
        train_ratio=0.8,
        random_seed=42,
        is_train=True,
        use_end_states=True,
        use_forces=True,
        use_resultants=True,
        chronology=False
    )
    
    # 打印第一个批次的信息
    batch_random = batched_dataset_random[0]
    print("\n随机批次信息:")
    for k, v in batch_random.items():
        if isinstance(v, torch.Tensor):
            print(f"  - {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  - {k}: {v}")
        else:
            print(f"  - {k}: {v}")
    
    # 比较不同模式的区别
    print(f"\n=== 模式比较 ===")
    print(f"完整数据时序模式序列数: {len(batched_dataset_chrono)}")
    print(f"仅动作序列模式序列数: {len(batched_dataset_action)}")
    print(f"随机采样模式序列数: {len(batched_dataset_random)}")
    print(f"时序模式是否连续: {batch_chrono['episode_info']['is_continuous']}")
    print(f"仅动作模式是否连续: {batch_action['episode_info']['is_continuous']}")
    print(f"随机模式是否连续: {batch_random['episode_info']['is_continuous']}")
    
    print(f"\n=== 数据大小比较 ===")
    print(f"完整模式 action shape: {batch_chrono['action'].shape}")
    print(f"仅动作模式 action shape: {batch_action['action'].shape}")
    print(f"完整模式是否包含end_states: {'end_states' in batch_chrono}")
    print(f"仅动作模式是否包含end_states: {'end_states' in batch_action}")
    
    # 演示便捷函数的使用
    print(f"\n=== 便捷函数演示 ===")
    train_ds, test_ds = create_train_test_datasets(
        data_root=data_root,
        categories=["cir_lar"],
        train_ratio=0.8,
        random_seed=42
    )
    print(f"训练集大小: {len(train_ds)}")
    print(f"测试集大小: {len(test_ds)}")
    
    train_batched, test_batched = create_train_test_batched_datasets(
        data_root=data_root,
        categories=["cir_lar"],
        sequence_length=5,
        stride=3,
        train_ratio=0.8,
        random_seed=42
    )
    print(f"训练批量数据集大小: {len(train_batched)}")
    print(f"测试批量数据集大小: {len(test_batched)}")
