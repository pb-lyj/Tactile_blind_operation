"""
新的策略学习数据集，支持时序和无时序模式
"""
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from glob import glob
import random


class FlexiblePolicyDataset(Dataset):
    """
    灵活的策略学习数据集，支持时序和无时序两种模式
    trajectories(文件夹级): 每个轨迹包含L帧数据
        action[L, 7] (timestep, dx, dy, dz, ...)
        other_data[L+1, ...] (end_states, forces, resultants等)
    """
    
    def __init__(self, data_root, categories=None, start_frame=0, train_ratio=0.8, random_seed=42, is_train=True, 
                 use_end_states=True, use_forces=True, use_resultants=True, normalize_config=None,
                 sequence_mode=False, sequence_length=10):
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
            normalize_config: 归一化配置字典
            sequence_mode: 是否使用时序模式
                - False: 无时序模式，每个样本是单个时间步的对齐数据，可以打乱
                - True: 有时序模式，每个样本是一个长度为sequence_length的序列片段，最后不满足sequence_length的片段丢弃
            sequence_length: 时序模式下的序列长度
        """
        self.data_root = data_root
        self.start_frame = start_frame
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.is_train = is_train
        self.use_end_states = use_end_states
        self.use_forces = use_forces
        self.use_resultants = use_resultants
        self.sequence_mode = sequence_mode
        self.sequence_length = sequence_length
        
        # 设置默认归一化配置
        if normalize_config is None:
            normalize_config = {
                'forces': 'zscore', 'actions': 'minmax', 'end_states': 'zscore', 'resultants': 'minmax'
            }
        self.normalize_config = normalize_config
        
        # 初始化轨迹数据和索引
        self._load_trajectory_metadata(categories)
        self._build_indices()
        self._print_dataset_info()

    def _load_trajectory_metadata(self, categories):
        """加载轨迹元数据"""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        if categories is None:
            categories = [
                "cir_lar", "cir_med", "cir_sma",
                "rect_lar", "rect_med", "rect_sma", 
                "tri_lar", "tri_med", "tri_sma"
            ]
        
        self.trajectories = []
        total_episodes = 0
        
        for category in categories:
            category_path = os.path.join(self.data_root, category)
            if not os.path.exists(category_path):
                print(f"⚠️  警告: 类别目录不存在: {category_path}")
                continue
            
            trajectory_dirs = sorted([
                d for d in os.listdir(category_path) 
                if os.path.isdir(os.path.join(category_path, d))
            ])
            
            total_episodes += len(trajectory_dirs)
            
            for traj_dir in trajectory_dirs:
                traj_path = os.path.join(category_path, traj_dir)
                
                # 检查数据完整性
                if not self._validate_trajectory(traj_path):
                    continue
                
                # 获取轨迹长度
                try:
                    action_data = np.load(os.path.join(traj_path, "_action.npy"))
                    traj_length = len(action_data) - self.start_frame
                    
                    if traj_length <= 0:
                        continue
                    
                    # 如果是时序模式，检查长度是否足够
                    if self.sequence_mode and traj_length < self.sequence_length:
                        continue
                    
                    self.trajectories.append({
                        'path': traj_path,
                        'category': category,
                        'length': traj_length,
                        'dir_name': traj_dir
                    })
                    
                except Exception as e:
                    print(f"⚠️  警告: 加载轨迹数据失败 {traj_path}: {e}")
                    continue
        
        # 划分训练/测试轨迹
        train_count = int(len(self.trajectories) * self.train_ratio)
        random.shuffle(self.trajectories)
        
        if self.is_train:
            self.trajectories = self.trajectories[:train_count]
        else:
            self.trajectories = self.trajectories[train_count:]

    def _validate_trajectory(self, traj_path):
        """验证轨迹数据完整性和对齐"""
        required_files = ["_action.npy"]
        if self.use_end_states:
            required_files.append("_end_states.npy")
        if self.use_forces:
            required_files.extend(["_forces_l.npy", "_forces_r.npy"])
        if self.use_resultants:
            required_files.extend([
                "_resultant_force_l.npy", "_resultant_force_r.npy",
                "_resultant_moment_l.npy", "_resultant_moment_r.npy"
            ])
        
        # 检查文件存在性
        for file in required_files:
            if not os.path.exists(os.path.join(traj_path, file)):
                return False
        
        # 检查数据对齐 - action有L帧，其他数据有L+1帧
        try:
            action_data = np.load(os.path.join(traj_path, "_action.npy"))
            action_length = len(action_data)
            
            # 检查其他数据长度应该是 action_length + 1
            other_files = []
            if self.use_end_states:
                other_files.append("_end_states.npy")
            if self.use_forces:
                other_files.extend(["_forces_l.npy", "_forces_r.npy"])
            if self.use_resultants:
                other_files.extend([
                    "_resultant_force_l.npy", "_resultant_force_r.npy",
                    "_resultant_moment_l.npy", "_resultant_moment_r.npy"
                ])
            
            for file in other_files:
                data = np.load(os.path.join(traj_path, file))
                if len(data) != action_length + 1:
                    print(f"⚠️  数据对齐错误 {traj_path}/{file}: "
                          f"期望长度{action_length + 1}, 实际长度{len(data)}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  数据验证失败 {traj_path}: {e}")
            return False

    def _build_indices(self):
        """构建数据索引"""
        self.indices = []
        
        if self.sequence_mode:
            # 时序模式：构建序列片段索引
            for traj_idx, traj_info in enumerate(self.trajectories):
                traj_length = traj_info['length']
                max_start = traj_length - self.sequence_length
                for start_idx in range(max_start + 1):
                    self.indices.append({
                        'traj_idx': traj_idx,
                        'start_idx': start_idx,
                        'end_idx': start_idx + self.sequence_length
                    })
        else:
            # 无时序模式：构建单步索引
            for traj_idx, traj_info in enumerate(self.trajectories):
                traj_length = traj_info['length']
                for step_idx in range(traj_length):
                    self.indices.append({
                        'traj_idx': traj_idx,
                        'step_idx': step_idx
                    })
            
            # 无时序模式下打乱索引
            random.shuffle(self.indices)

    def _print_dataset_info(self):
        """打印数据集信息"""
        total_trajectories = len(self.trajectories)
        total_samples = len(self.indices)
        
        print(f"[FlexiblePolicyDataset] 数据统计:")
        print(f"  - 模式: {'时序模式' if self.sequence_mode else '无时序模式'}")
        if self.sequence_mode:
            print(f"  - 序列长度: {self.sequence_length}")
        print(f"  - 轨迹数量: {total_trajectories}")
        print(f"  - 样本数量: {total_samples}")
        print(f"  - 当前集合: {'训练集' if self.is_train else '测试集'}")
        print(f"  - 数据类型: action" + 
              (", end_states" if self.use_end_states else "") +
              (", forces" if self.use_forces else "") + 
              (", resultants" if self.use_resultants else ""))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """获取单个样本"""
        index_info = self.indices[idx]
        traj_info = self.trajectories[index_info['traj_idx']]
        traj_path = traj_info['path']
        
        if self.sequence_mode:
            # 时序模式：返回序列片段
            start_idx = index_info['start_idx']
            end_idx = index_info['end_idx']
            return self._load_sequence_data(traj_path, traj_info, start_idx, end_idx)
        else:
            # 无时序模式：返回单个时间步
            step_idx = index_info['step_idx']
            return self._load_single_step_data(traj_path, traj_info, step_idx)

    def _load_single_step_data(self, traj_path, traj_info, step_idx):
        """无时序模式：加载单个时间步的对齐数据（简化版本 - 所有数据都是L帧）"""
        # 计算实际的数据索引
        actual_idx = self.start_frame + step_idx
        
        # 加载action数据
        action_data = np.load(os.path.join(traj_path, "_action.npy"))
        action = action_data[actual_idx, 1:]  # 去掉时间戳
        action = self._normalize_data(action, 'actions')
        
        result = {
            'action': torch.FloatTensor(action),
            'category': traj_info['category'],
            'trajectory_id': traj_info['dir_name'],
            'step_idx': step_idx
        }
        
        # 加载end_states数据（截取为L帧，抛弃最后一帧）
        if self.use_end_states:
            end_states_data = np.load(os.path.join(traj_path, "_end_states.npy"))
            # 只取前L帧，抛弃第L+1帧
            current_state = end_states_data[actual_idx, 1:]  # 当前状态
            
            current_state = self._normalize_data(current_state, 'end_states')
            result['current_state'] = torch.FloatTensor(current_state)
            result['timestamps'] = torch.FloatTensor([end_states_data[actual_idx, 0]])
        
        # 加载forces数据（截取为L帧，抛弃最后一帧）
        if self.use_forces:
            forces_l_data = np.load(os.path.join(traj_path, "_forces_l.npy"))
            forces_r_data = np.load(os.path.join(traj_path, "_forces_r.npy"))
            
            # 只取前L帧，抛弃第L+1帧
            forces_l = forces_l_data[actual_idx]
            forces_r = forces_r_data[actual_idx]
            
            forces_l = self._normalize_data(forces_l, 'forces')
            forces_r = self._normalize_data(forces_r, 'forces')
            
            result['forces_l'] = torch.FloatTensor(forces_l)
            result['forces_r'] = torch.FloatTensor(forces_r)
        
        # 加载resultants数据（截取为L帧，抛弃最后一帧）
        if self.use_resultants:
            resultant_force_l = np.load(os.path.join(traj_path, "_resultant_force_l.npy"))[actual_idx]
            resultant_force_r = np.load(os.path.join(traj_path, "_resultant_force_r.npy"))[actual_idx]
            resultant_moment_l = np.load(os.path.join(traj_path, "_resultant_moment_l.npy"))[actual_idx]
            resultant_moment_r = np.load(os.path.join(traj_path, "_resultant_moment_r.npy"))[actual_idx]
            
            resultant_force_l = self._normalize_data(resultant_force_l, 'resultants')
            resultant_force_r = self._normalize_data(resultant_force_r, 'resultants')
            resultant_moment_l = self._normalize_data(resultant_moment_l, 'resultants')
            resultant_moment_r = self._normalize_data(resultant_moment_r, 'resultants')
            
            result['resultant_force_l'] = torch.FloatTensor(resultant_force_l)
            result['resultant_force_r'] = torch.FloatTensor(resultant_force_r)
            result['resultant_moment_l'] = torch.FloatTensor(resultant_moment_l)
            result['resultant_moment_r'] = torch.FloatTensor(resultant_moment_r)
        
        return result

    def _load_sequence_data(self, traj_path, traj_info, start_idx, end_idx):
        """时序模式：加载序列片段数据（简化版本 - 所有数据都是L帧长度）"""
        # 计算实际的数据索引范围
        actual_start = self.start_frame + start_idx
        actual_end = self.start_frame + end_idx
        
        # 加载action序列
        action_data = np.load(os.path.join(traj_path, "_action.npy"))
        actions = action_data[actual_start:actual_end, 1:]  # 去掉时间戳
        actions = self._normalize_data(actions, 'actions')
        
        result = {
            'actions': torch.FloatTensor(actions),  # (L, action_dim)
            'category': traj_info['category'],
            'trajectory_id': traj_info['dir_name'],
            'start_idx': start_idx,
            'end_idx': end_idx,
            'sequence_length': end_idx - start_idx
        }
        
        # 加载end_states序列（截取为L帧，抛弃最后一帧）
        if self.use_end_states:
            end_states_data = np.load(os.path.join(traj_path, "_end_states.npy"))
            # 只取前L帧，与action对齐
            states = end_states_data[actual_start:actual_end, 1:]
            states = self._normalize_data(states, 'end_states')
            timestamps = end_states_data[actual_start:actual_end, 0]
            
            result['states'] = torch.FloatTensor(states)  # (L, state_dim)
            result['timestamps'] = torch.FloatTensor(timestamps)
        
        # 加载forces序列（截取为L帧，抛弃最后一帧）
        if self.use_forces:
            forces_l_data = np.load(os.path.join(traj_path, "_forces_l.npy"))
            forces_r_data = np.load(os.path.join(traj_path, "_forces_r.npy"))
            
            # 只取前L帧，与action对齐
            forces_l = forces_l_data[actual_start:actual_end]
            forces_r = forces_r_data[actual_start:actual_end]
            
            forces_l = self._normalize_data(forces_l, 'forces')
            forces_r = self._normalize_data(forces_r, 'forces')
            
            result['forces_l'] = torch.FloatTensor(forces_l)  # (L, 3, 20, 20)
            result['forces_r'] = torch.FloatTensor(forces_r)  # (L, 3, 20, 20)
        
        # 加载resultants序列（截取为L帧，抛弃最后一帧）
        if self.use_resultants:
            resultant_force_l = np.load(os.path.join(traj_path, "_resultant_force_l.npy"))[actual_start:actual_end]
            resultant_force_r = np.load(os.path.join(traj_path, "_resultant_force_r.npy"))[actual_start:actual_end]
            resultant_moment_l = np.load(os.path.join(traj_path, "_resultant_moment_l.npy"))[actual_start:actual_end]
            resultant_moment_r = np.load(os.path.join(traj_path, "_resultant_moment_r.npy"))[actual_start:actual_end]
            
            resultant_force_l = self._normalize_data(resultant_force_l, 'resultants')
            resultant_force_r = self._normalize_data(resultant_force_r, 'resultants')
            resultant_moment_l = self._normalize_data(resultant_moment_l, 'resultants')
            resultant_moment_r = self._normalize_data(resultant_moment_r, 'resultants')
            
            result['resultant_force_l'] = torch.FloatTensor(resultant_force_l)  # (L, 3)
            result['resultant_force_r'] = torch.FloatTensor(resultant_force_r)  # (L, 3)
            result['resultant_moment_l'] = torch.FloatTensor(resultant_moment_l)  # (L, 3)
            result['resultant_moment_r'] = torch.FloatTensor(resultant_moment_r)  # (L, 3)
        
        return result

    def _normalize_data(self, data, data_type):
        """数据归一化处理"""
        normalize_method = self.normalize_config.get(data_type, None)
        
        if normalize_method is None:
            return data
        
        if normalize_method == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            if std > 1e-8:
                return (data - mean) / std
            else:
                return data - mean
                
        elif normalize_method == 'minmax':
            data_min = np.min(data)
            data_max = np.max(data)
            
            if not hasattr(self, 'normalization_params'):
                self.normalization_params = {}
            self.normalization_params[data_type] = {
                'min': data_min, 'max': data_max, 'method': 'minmax'
            }
            
            if data_max - data_min > 1e-8:
                return 2.0 * (data - data_min) / (data_max - data_min) - 1.0
            else:
                return np.zeros_like(data)
        
        return data

    def denormalize_data(self, normalized_data, data_type):
        """反归一化"""
        if not hasattr(self, 'normalization_params') or data_type not in self.normalization_params:
            return normalized_data
        
        params = self.normalization_params[data_type]
        if params['method'] == 'minmax':
            data_min = params['min']
            data_max = params['max']
            return (normalized_data + 1.0) / 2.0 * (data_max - data_min) + data_min
        
        return normalized_data


def create_flexible_datasets(data_root, categories=None, train_ratio=0.8, random_seed=42,
                           start_frame=0, use_end_states=True, use_forces=True, use_resultants=True,
                           normalize_config=None, sequence_mode=False, sequence_length=10):
    """
    创建灵活的训练集和测试集
    """
    train_dataset = FlexiblePolicyDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        train_ratio=train_ratio,
        random_seed=random_seed,
        is_train=True,
        use_end_states=use_end_states,
        use_forces=use_forces,
        use_resultants=use_resultants,
        normalize_config=normalize_config,
        sequence_mode=sequence_mode,
        sequence_length=sequence_length
    )
    
    test_dataset = FlexiblePolicyDataset(
        data_root=data_root,
        categories=categories,
        start_frame=start_frame,
        train_ratio=train_ratio,
        random_seed=random_seed,
        is_train=False,
        use_end_states=use_end_states,
        use_forces=use_forces,
        use_resultants=use_resultants,
        normalize_config=normalize_config,
        sequence_mode=sequence_mode,
        sequence_length=sequence_length
    )
    
    return train_dataset, test_dataset
