import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.ddpm_refine import DiffusionRefine
# from utils.metrics import calculate_psnr, calculate_ssim, calculate_lpips
from torchvision.utils import save_image
import numpy as np

def load_image(path):
    return transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)

@torch.no_grad()
def sample_residual(model, I0, I1, I_pred, steps=50):
    model.eval()
    B, C, H, W = I_pred.shape
    device = I0.device
    x = torch.randn_like(I_pred)

    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t] * B, device=device)
        motion = torch.abs(I1 - I0)
        cond = torch.cat([I0, I1, I_pred, motion], dim=1)
        x_input = torch.cat([cond, x], dim=1)
        eps_theta = model.model(x_input, t_tensor)

        alpha_t = model.alphas[t]
        alpha_hat_t = model.alpha_hat[t]
        beta_t = model.betas[t]

        x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_hat_t)) * eps_theta)
        if t > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = x + sigma_t * noise

    return x

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionRefine(timesteps=args.timesteps)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    # model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Resize((256, 448))
    triplet_dirs = sorted(os.listdir(args.data_root))
    # psnr_list, ssim_list, lpips_list = [], [], []

    # os.makedirs(args.save_dir, exist_ok=True)

    for folder in triplet_dirs:
        folder_path = os.path.join(args.data_root, folder)
        I0 = load_image(os.path.join(folder_path, "I0.png")).to(device)
        I1 = load_image(os.path.join(folder_path, "I1.png")).to(device)
        I_pred = load_image(os.path.join(folder_path, "I_pred.png")).to(device)
        I_gt = load_image(os.path.join(folder_path, "I_gt.png")).to(device)

        residual = sample_residual(model, I0, I1, I_pred, steps=args.sample_steps)
        I_refined = (I_pred + residual).clamp(0, 1)

        # Save output
        if args.save_images:
            out_path = os.path.join(folder_path, "refined.png")
            save_image(I_refined, out_path)

        # psnr = calculate_psnr(I_refined, I_gt)
        # ssim = calculate_ssim(I_refined, I_gt)
        # lpips_val = calculate_lpips(I_refined, I_gt)

        # psnr_list.append(psnr)
        # ssim_list.append(ssim)
        # lpips_list.append(lpips_val)

        # print(f"[{folder}] PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | LPIPS: {lpips_val:.4f}")

    # print("=== Final Avg Results ===")
    # print(f"PSNR: {np.mean(psnr_list):.2f}")
    # print(f"SSIM: {np.mean(ssim_list):.4f}")
    # print(f"LPIPS: {np.mean(lpips_list):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Refiner")
    parser.add_argument('--data_root', type=str, default='../fast_motion_triplets')
    # parser.add_argument('--save_dir', type=str, default='./eval_outputs')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sample_steps', type=int, default=50)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--save_images', action='store_true')
    args = parser.parse_args()

    evaluate(args)
