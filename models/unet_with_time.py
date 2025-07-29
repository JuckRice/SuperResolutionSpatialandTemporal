import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

class UNetWithTimeEmbedding(nn.Module):
    def __init__(self, in_channels=9, out_channels=3, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        def block(in_feat, out_feat):
            return nn.Sequential(
                nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.down1 = block(in_channels, 64)
        self.down2 = block(64, 128)
        self.down3 = block(128, 256)
        self.middle = block(256, 256)
        self.up3 = block(256 + 256, 128)
        self.up2 = block(128 + 128, 64)
        self.up1 = block(64 + 64, out_channels)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.time_to_feat = nn.Linear(time_emb_dim, 256)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        t_feat = self.time_to_feat(t_emb).unsqueeze(-1).unsqueeze(-1)

        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        mid = self.middle(self.pool(d3)) + t_feat

        u3 = self.up3(torch.cat([self.upsample(mid), d3], dim=1))
        u2 = self.up2(torch.cat([self.upsample(u3), d2], dim=1))
        out = self.up1(torch.cat([self.upsample(u2), d1], dim=1))

        return out
