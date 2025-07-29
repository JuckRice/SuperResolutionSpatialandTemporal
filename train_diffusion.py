import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.ddpm_refine import DiffusionRefine
from datasets.refine_dataset import RefineNetDataset
from torchvision import transforms
import os
from losses.refineloss import CombinedLoss


parser = argparse.ArgumentParser(description="Train Diffusion Refiner for Frame Interpolation")

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--save_dir', type=str, default='./checkpoints/diffusion_default')
parser.add_argument('--save_freq', type=int, default=50, help="Checkpoint saving frequency")
parser.add_argument('--data_root', type=str, default='../fast_motion_triplets')

parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--timesteps', type=int, default=1000, help="Diffusion time steps")


criterion = CombinedLoss(w_c=1.0, w_p=0.1, w_e=0.05)

def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 448)),
        transforms.ToTensor()
    ])
    train_dataset = RefineNetDataset(root_dir=args.data_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = DiffusionRefine(timesteps=args.timesteps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for I0, I1, I_pred, I_gt in train_loader:
            I0, I1, I_pred, I_gt = I0.to(device), I1.to(device), I_pred.to(device), I_gt.to(device)
            residual = I_gt - I_pred

            optimizer.zero_grad()
            loss = model(I0, I1, I_pred, residual)  # diffusion 模型负责损失计算和时间采样
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch:03d}] Total Loss: {avg_loss:.4f}")

        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.save_dir, f"diffusion_epoch{epoch:03d}.pth"))
            print(f"✅ Saved checkpoint at epoch {epoch:03d}")

if __name__ == "__main__":
    main()
