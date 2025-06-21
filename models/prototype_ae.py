import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import os

class TactileDataset(Dataset):
    """统一的触觉数据集加载器
    
    模式:
        - 'train': 用于原型发现的训练模式
        - 'validate': 用于原型验证的多环境模式
        - 'episode': 用于APT的情节批次模式
        
        # 训练模式
        train_dataset = TactileDataset(
            root_dir="./organized_data_1_1/env_0",
            mode='train'
        )

        # 验证模式
        validate_dataset = TactileDataset(
            root_dir="./organized_data_1_1",
            mode='validate',
            exclude_envs=["env_0"]
        )

        # 情节模式
        episode_dataset = TactileDataset(
            root_dir="./organized_data_1_1",
            mode='episode',
            action_root="./organized_data_1_1",
            min_episode_len=35,
            max_episode_len=200
        )
    """
    def __init__(self, root_dir, mode='train', exclude_envs=None, **kwargs):
        """
        Args:
            root_dir: 数据根目录
            mode: 'train', 'validate', 或 'episode'
            exclude_envs: 要排除的环境列表
            **kwargs: 其他参数
                - action_root: episode模式下的动作数据目录
                - min_episode_len: episode模式下最小情节长度 (默认35)
                - max_episode_len: episode模式下最大情节长度 (默认200)
        """
        self.mode = mode
        self.samples = []
        
        if mode == 'train':
            self._init_train_mode(root_dir)
        elif mode == 'validate':
            self._init_validate_mode(root_dir, exclude_envs)
        elif mode == 'episode':
            self._init_episode_mode(root_dir, kwargs.get('action_root'),
                                  kwargs.get('min_episode_len', 35),
                                  kwargs.get('max_episode_len', 200))
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    def _init_train_mode(self, root_dir):
        """训练模式初始化: 加载单一环境的所有触觉数据"""
        paths = sorted(glob(os.path.join(root_dir, "episode_*", "tactile.npy")))
        samples = []
        
        for path in paths:
            data = np.load(path)  # shape (T, 6, H, W)
            for t in range(data.shape[0]):
                frame = data[t, -1]
                samples.append(frame[0:3])  # sensor 1
                samples.append(frame[3:6])  # sensor 2
                
        self.samples = np.stack(samples)
        self.samples[:, 0:2] /= 0.1  # normalize XY range
        
    def _init_validate_mode(self, root_dir, exclude_envs):
        """验证模式初始化: 多环境加载带路径信息"""
        exclude_envs = set(exclude_envs or [])
        self.sample_index = []  # 每项是 (path, t, sensor_id)
        
        env_dirs = sorted(glob(os.path.join(root_dir, "env_*")))
        env_dirs = [env for env in env_dirs if os.path.basename(env) not in exclude_envs]
        
        for env in env_dirs:
            episode_dirs = sorted(glob(os.path.join(env, "episode_*")))
            for ep in episode_dirs:
                tactile_path = os.path.join(ep, "tactile.npy")
                if os.path.exists(tactile_path):
                    try:
                        data = np.load(tactile_path, mmap_mode='r')
                        T = data.shape[0]
                        for t in range(T):
                            self.sample_index.append((tactile_path, t, 0))  # sensor_left
                            self.sample_index.append((tactile_path, t, 1))  # sensor_right
                    except Exception as e:
                        print(f"Warning: Could not load {tactile_path}: {e}")
                        
    def _init_episode_mode(self, root_dir, action_root, min_len=35, max_len=200):
        """情节模式初始化: 加载完整情节序列和对应动作"""
        self.episodes = []  # [(z_seq, a_seq, a_next, z_next), ...]
        
        env_dirs = sorted(glob(os.path.join(root_dir, "env_*")))
        for env in env_dirs:
            env_name = os.path.basename(env)
            act_paths = sorted(glob(os.path.join(action_root, env_name, "episode_*", "action.npy")))
            
            for act_path in act_paths:
                ep_dir = os.path.dirname(act_path)
                ep_name = os.path.basename(ep_dir)
                act_data = np.load(act_path)
                
                # 加载触觉数据
                tactile_path = os.path.join(ep_dir, "tactile.npy")
                if not os.path.exists(tactile_path):
                    print(f"Missing tactile: {tactile_path}")
                    continue
                    
                z_data = self._load_and_process_tactile(tactile_path)
                if z_data is None or len(act_data) < min_len or len(act_data) > max_len:
                    continue
                
                self._process_episode(z_data, act_data)
                
    def _load_and_process_tactile(self, path):
        """加载并预处理触觉数据"""
        try:
            data = np.load(path)
            processed = []
            for t in range(data.shape[0]):
                frame = data[t, -1]
                left = frame[0:3].copy()
                right = frame[3:6].copy()
                left[0:2] /= 0.1
                right[0:2] /= 0.1
                processed.append(np.concatenate([left, right]))
            return np.stack(processed)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            return None
            
    def _process_episode(self, z_data, act_data):
        """处理单个情节的数据"""
        for t in range(30, len(act_data) - 1):
            z_seq = z_data[:t]
            a_seq = act_data[:t]
            a_next = act_data[t]
            z_next = z_data[t]
            self.episodes.append((z_seq, a_seq, a_next, z_next))
            
    def __len__(self):
        if self.mode == 'validate':
            return len(self.sample_index)
        elif self.mode == 'episode':
            return len(self.episodes)
        else:
            return len(self.samples)
            
    def __getitem__(self, idx):
        if self.mode == 'train':
            return torch.tensor(self.samples[idx], dtype=torch.float32)
        elif self.mode == 'validate':
            path, t, sensor_id = self.sample_index[idx]
            data = np.load(path)
            frame = data[t, -1]
            sample = frame[0:3] if sensor_id == 0 else frame[3:6]
            sample[0:2] /= 0.1
            return (
                torch.tensor(sample, dtype=torch.float32),
                (path, torch.tensor(t), torch.tensor(sensor_id))
            )
        else:  # episode mode
            return self.episodes[idx]
        
# ==================== Spatial Transformer Module ====================
class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 6)
        )
        self.localization[-1].weight.data.zero_()
        self.localization[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.localization(x)  # (B, 6)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_trans = F.grid_sample(x, grid, align_corners=True)
        return x_trans, theta

# 添加共享 STN 网络的实现
class SharedSpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享的特征提取卷积层
        self.shared_features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # 共享的FC层
        self.shared_fc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 32),
            nn.ReLU()
        )
        
        # 每个原型独立的最终变换层
        self.loc_heads = nn.ModuleList()
        
    def add_localization_head(self):
        # 只保留最后一层作为独立的变换层
        head = nn.Linear(32, 6)
        # 初始化为恒等变换
        head.weight.data.zero_()
        head.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        self.loc_heads.append(head)
        
    def forward(self, x, proto_idx):
        # 共享特征提取
        features = self.shared_features(x)
        # 共享FC层
        features = self.shared_fc(features)
        # 独立变换层
        theta = self.loc_heads[proto_idx](features)
        theta = theta.view(-1, 2, 3)
        
        # 空间变换
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x_trans = F.grid_sample(x, grid, align_corners=True)
        return x_trans, theta

# ==================== Prototype Autoencoder ====================
class PrototypeAutoencoder(nn.Module):
    def __init__(self, num_prototypes=16, input_shape=(3, 32, 32), share_stn=True):
        super().__init__()
        self.K = num_prototypes
        self.H, self.W = input_shape[1:]
        self.share_stn = share_stn

        # 可学习的原型库
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, *input_shape))

        # STN 模块
        if share_stn:
            self.shared_stn = SharedSpatialTransformer()
            # 为每个原型添加定位头
            for _ in range(num_prototypes):
                self.shared_stn.add_localization_head()
        else:
            self.stn_modules = nn.ModuleList([SpatialTransformer() for _ in range(num_prototypes)])

        # CNN encoder
        self.final_linear = nn.Linear(32, num_prototypes)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            self.final_linear,
            # nn.Softmax(dim=-1)  # output ω: mode activation weights
            nn.Sigmoid()
        )
        
        # 初始化偏置
        nn.init.xavier_uniform_(self.final_linear.weight)
        self.final_linear.bias.data.fill_(2.0)

    def forward(self, x):
        weights = self.encoder(x)  # (B, K)
        weights = torch.clamp(weights, 1e-4, 1 - 1e-4)
        B = x.size(0)
        protos = self.prototypes.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, K, C, H, W)

        transformed_protos = []
        thetas = []

        for k in range(self.K):
            proto_k = protos[:, k]  # (B, C, H, W)
            if self.share_stn:
                trans_k, theta_k = self.shared_stn(proto_k, k)
            else:
                trans_k, theta_k = self.stn_modules[k](proto_k)
            transformed_protos.append(trans_k)
            thetas.append(theta_k)

        transformed_protos = torch.stack(transformed_protos, dim=1)  # (B, K, C, H, W)
        thetas = torch.stack(thetas, dim=1)  # (B, K, 2, 3)

        weights_exp = weights.view(B, self.K, 1, 1, 1)
        recon = (weights_exp * transformed_protos).sum(dim=1)  # (B, C, H, W)

        return recon, weights, transformed_protos, thetas

# ==================== Loss Function ====================
def compute_losses(x, recon, weights, transformed_protos, thetas,
                   diversity_lambda=0.1, entropy_lambda=10.0, stn_reg_lambda=0.05):

    # Reconstruction loss
    recon_loss = F.mse_loss(recon, x)

    # Diversity loss via pairwise similarity
    B, K, C, H, W = transformed_protos.shape
    avg_protos = transformed_protos.mean(dim=0)  # (K, C, H, W)
    normed_protos = F.normalize(avg_protos.view(K, -1), dim=1)
    similarity_matrix = torch.matmul(normed_protos, normed_protos.T)
    off_diag = similarity_matrix - torch.eye(K, device=similarity_matrix.device)
    diversity_loss = (off_diag ** 2).mean()

    # Sparsity loss
    # sparsity_loss = weights.mean()  # 惩罚平均激活

    # Entropy loss on weights
    binary_entropy = - (weights * torch.log(weights + 1e-8) + (1 - weights) * torch.log(1 - weights + 1e-8))
    entropy_loss = entropy_lambda * binary_entropy.mean()


    # STN regularization loss
    identity = torch.tensor([[1., 0., 0.], [0., 1., 0.]], device=thetas.device).view(1, 1, 2, 3)
    stn_loss = F.mse_loss(thetas, identity.expand_as(thetas))

    # Final loss
    total_loss = recon_loss + \
                 diversity_lambda * diversity_loss + \
                 entropy_loss + \
                 stn_reg_lambda * stn_loss

    return total_loss, {
        "recon_loss": recon_loss.item(),
        "diversity_loss": diversity_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "stn_loss": stn_loss.item(),
        "total_loss": total_loss.item()
    }

