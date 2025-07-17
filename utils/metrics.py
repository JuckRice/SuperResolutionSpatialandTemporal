import torch
import torch.nn.functional as F
import lpips
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import numpy as np

lpips_alex = lpips.LPIPS(net='alex')
lpips_alex.eval()

def calculate_psnr(img1, img2):
    img1_np = img1.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return compare_psnr(img1_np, img2_np, data_range=1.0)

def calculate_ssim(img1, img2):
    img1_np = img1.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    img2_np = img2.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
    return compare_ssim(img1_np, img2_np, multichannel=True, data_range=1.0)

def calculate_lpips(img1, img2):
    with torch.no_grad():
        dist = lpips_alex(img1, img2)
    return dist.item()
