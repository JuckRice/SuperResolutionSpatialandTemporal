import torch
import torch.nn as nn

def charbonnier_loss(pred, gt, eps=1e-6):
    diff = pred - gt
    loss = torch.sqrt(diff * diff + eps)
    return loss.mean()

# 可扩展：添加 perceptual_loss, edge_loss 等
