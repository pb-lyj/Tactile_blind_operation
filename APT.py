## Action Prototype Time Series (APT) Module
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
from tactile_clustering.Physical_mapping import Logger

# ============ Config ============
ACTIVATION_ROOT = "validation_results/activations_1_1"
ACTION_ROOT = "organized_data_1_1"
ENV_LIST = ["env_1", "env_2", "env_3", "env_4", "env_5", "env_6", "env_7", "env_8", "env_9", "env_10"]
BATCH_SIZE = 256
EPOCHS = 100
LR = 1e-3
ACTION_LOSS_WEIGHT = 1.0
Z_LOSS_WEIGHT = 1.0

# ============ Dataset ============
class EpisodeBatchDataset(Dataset):
    def __init__(self):
        self.episodes = []

        for env in ENV_LIST:
            act_paths = sorted(glob(os.path.join(ACTION_ROOT, env, "episode_*", "action.npy")))
            for act_path in act_paths:
                ep_dir = os.path.dirname(act_path)
                ep_name = os.path.basename(ep_dir)
                act_data = np.load(act_path)

                z_path = os.path.join(ACTIVATION_ROOT, env, f"{ep_name}_activations.npy")
                if not os.path.exists(z_path):
                    print(f"Missing activation: {z_path}")
                    continue

                # 从30条后用于训练，保证触觉密集
                z_data = np.load(z_path)  # shape (2*T, 64)
                if z_data.shape[0] != 2 * len(act_data) or len(act_data) < 35 or len(act_data) > 200:
                    print(f"Skipped: {ep_name} due to mismatched length {z_data.shape[0]} vs {len(act_data)}")
                    continue

                z_seq_batch, a_seq_batch, a_next_batch, z_next_batch = [], [], [], []
                for t in range(30, len(act_data) - 1):
                    z_seq = np.stack([np.concatenate([z_data[2*i], z_data[2*i+1]]) for i in range(t)], axis=0)
                    a_seq = act_data[:t]
                    a_next = act_data[t]
                    z_next = np.concatenate([z_data[2*t], z_data[2*t+1]])
                    z_seq_batch.append(z_seq)
                    a_seq_batch.append(a_seq)
                    a_next_batch.append(a_next)
                    z_next_batch.append(z_next)

                self.episodes.append((z_seq_batch, a_seq_batch, a_next_batch, z_next_batch))

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]


# ============ Model ============
class PrototypeToActionModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = nn.LSTM(input_size=131, hidden_size=hidden_dim, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, z_seq, a_seq):
        x = torch.cat([z_seq, a_seq], dim=-1)  # (B, T, 131)
        out, _ = self.encoder(x)
        h = out[:, -1]
        return self.policy_head(h), self.transition_head(h)


# ============ Training ============
def cosine_and_magnitude_loss(pred, target):
    cosine_loss = 1 - F.cosine_similarity(pred, target).mean()
    magnitude_loss = F.mse_loss(pred, target)
    return cosine_loss, magnitude_loss

def train():
    dataset = EpisodeBatchDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, collate_fn=lambda x: x[0])
    model = PrototypeToActionModel().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        sample_count = 0
        for z_seq_batch, a_seq_batch, a_next_batch, z_next_batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            for z_seq, a_seq, a_next, z_next in zip(z_seq_batch, a_seq_batch, a_next_batch, z_next_batch):
                max_len = 10
                t_len = len(z_seq)
                if t_len > max_len:
                    z_seq = z_seq[-max_len:]
                    a_seq = a_seq[-max_len:]
                else:
                    pad_len = max_len - t_len
                    z_seq = np.pad(z_seq, ((pad_len, 0), (0, 0)))
                    a_seq = np.pad(a_seq, ((pad_len, 0), (0, 0)))

                z_seq = torch.tensor(z_seq, dtype=torch.float32).unsqueeze(0).cuda()
                a_seq = torch.tensor(a_seq, dtype=torch.float32).unsqueeze(0).cuda()
                a_next = torch.tensor(a_next, dtype=torch.float32).unsqueeze(0).cuda()
                z_next = torch.tensor(z_next, dtype=torch.float32).unsqueeze(0).cuda()

                pred_a, pred_z = model(z_seq, a_seq)
                cosine_loss, magnitude_loss = cosine_and_magnitude_loss(pred_a, a_next)
                action_loss = 0.5 * cosine_loss + 0.5 * magnitude_loss
                z_loss = F.mse_loss(pred_z, z_next)
                loss = ACTION_LOSS_WEIGHT * action_loss + Z_LOSS_WEIGHT * z_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                sample_count += 1

        print(f"Epoch {epoch+1}, Loss: {total_loss / sample_count:.4f}")

    torch.save(model.state_dict(), "action_predictor.pt")
    
    sys.stdout = Logger("APT train.log")
    print("=" * 50)
    print("Configuration:")
    print(f"ACTIVATION_ROOT  = {ACTIVATION_ROOT}")
    print(f"ACTION_ROOT      = {ACTION_ROOT}")
    print(f"ENV_LIST         = {ENV_LIST}")
    print(f"BATCH_SIZE       = {BATCH_SIZE}")
    print(f"EPOCHS           = {EPOCHS}")
    print(f"LEARNING_RATE    = {LR}")
    print(f"ACTION_LOSS_WT   = {ACTION_LOSS_WEIGHT}")
    print(f"Z_LOSS_WT        = {Z_LOSS_WEIGHT}")
    print("=" * 50 + "\n")
    print("Training complete. Model saved at Program ROOT\n")
    print(f"Loss:Loss: {total_loss / sample_count:.4f}\n")
    print(f"cosine_loss:{cosine_loss:.4f}, magnitude_loss:{magnitude_loss:.4f}, z_loss:{z_loss:.4f}")


if __name__ == '__main__':
    train()
