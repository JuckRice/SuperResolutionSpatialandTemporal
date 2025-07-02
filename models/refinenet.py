import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinementNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(9, 64, 3, padding=1),  # Input Chns: I0 + I1 + I_pred
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)   # Refine Residual
        )

    def forward(self, I0, I1, I_pred):
        x = torch.cat([I0, I1, I_pred], dim=1)
        delta = self.model(x)
        return I_pred + delta  # Refined Output
