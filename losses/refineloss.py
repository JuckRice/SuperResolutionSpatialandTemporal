import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16

# ========================
# 1. Charbonnier Loss
# ========================
def charbonnier_loss(pred, gt, eps=1e-6):
    diff = pred - gt
    loss = torch.sqrt(diff * diff + eps)
    return loss.mean()

# ========================
# 2. Perceptual Loss (VGG)
# ========================
class PerceptualLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8], use_cuda=True):
        super().__init__()
        vgg = vgg16(pretrained=True).features[:max(layer_ids) + 1]
        self.vgg = vgg.eval()
        if use_cuda:
            self.vgg = self.vgg.cuda()
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.layer_ids = layer_ids
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            if i in self.layer_ids:
                loss += F.l1_loss(x, y)
        return loss

# ========================
# 3. Edge Loss (Sobel)
# ========================
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]).float()
        sobel_y = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]).float()
        # 扩展为3通道
        self.weight_x = nn.Parameter(sobel_x.repeat(3, 1, 1, 1), requires_grad=False)
        self.weight_y = nn.Parameter(sobel_y.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, x, y):
        # x, y: [B, 3, H, W]
        x_edge = F.conv2d(x, self.weight_x.to(x.device), padding=1, groups=3) + F.conv2d(x, self.weight_y.to(x.device), padding=1, groups=3)
        y_edge = F.conv2d(y, self.weight_x.to(y.device), padding=1, groups=3) + F.conv2d(y, self.weight_y.to(y.device), padding=1, groups=3)
        return torch.mean(torch.sqrt((x_edge - y_edge)**2 + 1e-6))

# ========================
# 4. Combined Loss
# ========================
class CombinedLoss(nn.Module):
    def __init__(self, w_c=1.0, w_p=0.1, w_e=0.05):
        super().__init__()
        self.charbonnier = charbonnier_loss
        self.perceptual = PerceptualLoss()
        self.edge = EdgeLoss()
        self.w_c = w_c
        self.w_p = w_p
        self.w_e = w_e

    def forward(self, pred, gt):
        loss = self.w_c * self.charbonnier(pred, gt)
        loss += self.w_p * self.perceptual(pred, gt)
        loss += self.w_e * self.edge(pred, gt)
        return loss