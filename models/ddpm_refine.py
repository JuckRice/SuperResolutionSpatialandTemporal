import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=15, base_channels=64, time_embed_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, base_channels),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, x, t):
        t_embed = self.time_mlp(t).view(-1, 64, 1, 1)  # reshape for broadcast
        h = F.relu(self.conv1(x) + t_embed)
        h = F.relu(self.conv2(h) + t_embed)
        h = F.relu(self.conv3(h) + t_embed)
        out = self.conv_out(h)
        return out

# ========================
# Diffusion Refiner Model
# ========================
class DiffusionRefine(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.model = SimpleUNet()
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alphas", 1. - self.betas)
        self.register_buffer("alpha_hat", torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_start, t, noise=None):
        """Forward noising process"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1).sqrt()
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hat[t]).view(-1, 1, 1, 1).sqrt()
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise, noise

    def forward(self, I0, I1, I_pred, residual_gt):
        B, C, H, W = I0.size()
        t = torch.randint(0, self.timesteps, (B,), device=I0.device).long()
        x_noisy, noise = self.q_sample(residual_gt, t)

        # Construct condition input
        motion = torch.abs(I1 - I0)
        cond = torch.cat([I0, I1, I_pred, motion], dim=1)  # [B, 10, H, W]
        x_input = torch.cat([cond, x_noisy], dim=1)         # [B, 13, H, W]

        pred_noise = self.model(x_input, t)
        return F.mse_loss(pred_noise, noise)

# ========================
# Diffusion Knowledge Distillation Refiner
# ========================
class DiffusionKDRefine(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps
        self.teacher_net = SimpleUNet(in_channels=15)  # I0, I1, I_gt, motion, noise
        self.student_net = SimpleUNet(in_channels=15)  # I0, I1, I_pred, motion, noise

        self.register_buffer("betas", torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer("alphas", 1. - self.betas)
        self.register_buffer("alpha_hat", torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_hat = self.alpha_hat[t].view(-1, 1, 1, 1).sqrt()
        sqrt_one_minus_alpha_hat = (1 - self.alpha_hat[t]).view(-1, 1, 1, 1).sqrt()
        return sqrt_alpha_hat * x_start + sqrt_one_minus_alpha_hat * noise, noise

    def forward(self, I0, I1, I_pred, I_gt):
        B = I0.size(0)
        device = I0.device
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        # Residuals
        res_gt = I_gt - I_pred
        noisy_res, noise = self.q_sample(res_gt, t)

        # Motion encoding
        motion = torch.abs(I1 - I0)

        # Teacher input: I0, I1, I_gt, motion, noisy residual
        teacher_input = torch.cat([I0, I1, I_gt, motion, noisy_res], dim=1)
        teacher_pred_noise = self.teacher_net(teacher_input, t)

        # Student input: I0, I1, I_pred, motion, noisy residual
        student_input = torch.cat([I0, I1, I_pred, motion, noisy_res], dim=1)
        student_pred_noise = self.student_net(student_input, t)

        # Loss: Distillation
        kd_loss = F.mse_loss(student_pred_noise, teacher_pred_noise.detach())
        teacher_self_loss = F.mse_loss(teacher_pred_noise, noise)

        return kd_loss + teacher_self_loss, kd_loss.item(), teacher_self_loss.item()
