# prototype_baseline.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== Baseline Prototype Autoencoder ====================
class PrototypeAEBaseline(nn.Module):
    def __init__(self, num_prototypes=16, input_shape=(3, 32, 32)):
        super().__init__()
        self.K = num_prototypes
        self.C, self.H, self.W = input_shape

        # Learnable prototype library
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, self.C, self.H, self.W))

        # CNN encoder to output weights (ω)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.C, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, num_prototypes),
            # nn.Sigmoid()  # or use Softmax(dim=-1) if needed
            nn.Softmax(dim=-1)  # Output weights for each prototype
        )

        # Optional: bias initialization to encourage soft activation
        if isinstance(self.encoder[-1], nn.Linear):
            nn.init.xavier_uniform_(self.encoder[-1].weight)
            self.encoder[-1].bias.data.fill_(2.0)

    def forward(self, x):
        weights = self.encoder(x)  # (B, K)
        weights = torch.clamp(weights, 1e-4, 1 - 1e-4)
        B = x.size(0)

        # Expand prototype tensor
        protos = self.prototypes.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, K, C, H, W)
        weights_exp = weights.view(B, self.K, 1, 1, 1)

        # Weighted reconstruction
        recon = (weights_exp * protos).sum(dim=1)  # (B, C, H, W)
        return recon, weights, protos

# ==================== Loss Function ====================
def compute_baseline_losses(x, recon, weights, protos, diversity_lambda=0.1, entropy_lambda=10.0):
    # Reconstruction loss
    recon_loss = F.mse_loss(recon, x)

    # Diversity loss between prototypes
    normed_protos = F.normalize(protos.mean(dim=0).view(protos.size(1), -1), dim=1)  # (K, D)
    sim_matrix = torch.matmul(normed_protos, normed_protos.T)
    off_diag = sim_matrix - torch.eye(protos.size(1), device=sim_matrix.device)
    diversity_loss = (off_diag ** 2).mean()

    # Entropy across prototypes for each sample (batch mean)
    binary_entropy = - (weights * torch.log(weights + 1e-8) + (1 - weights) * torch.log(1 - weights + 1e-8))
    entropy_loss = entropy_lambda * binary_entropy.mean()

    total_loss = recon_loss + diversity_lambda * diversity_loss + entropy_loss
    return total_loss, {
        "recon_loss": recon_loss.item(),
        "diversity_loss": diversity_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "total_loss": total_loss.item()
    }
