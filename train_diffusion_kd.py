import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from models.ddpm_refine import DiffusionKDRefine
from datasets.refine_dataset import RefineNetDataset

parser = argparse.ArgumentParser(description="Train KD-based Diffusion Refiner for Frame Interpolation")

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--timesteps', type=int, default=1000)

parser.add_argument('--data_root', type=str, default='../fast_motion_triplets')
parser.add_argument('--save_dir', type=str, default='./checkpoints/diff_kd')
parser.add_argument('--save_freq', type=int, default=50)

def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 448)),
        transforms.ToTensor()
    ])
    dataset = RefineNetDataset(root_dir=args.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = DiffusionKDRefine(timesteps=args.timesteps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_kd = 0.0
        total_teacher = 0.0

        for I0, I1, I_pred, I_gt in dataloader:
            I0, I1, I_pred, I_gt = I0.to(device), I1.to(device), I_pred.to(device), I_gt.to(device)

            optimizer.zero_grad()
            loss, kd_loss_val, teacher_loss_val = model(I0, I1, I_pred, I_gt)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_kd += kd_loss_val
            total_teacher += teacher_loss_val

        avg_loss = total_loss / len(dataloader)
        avg_kd = total_kd / len(dataloader)
        avg_teacher = total_teacher / len(dataloader)

        print(f"[Epoch {epoch:03d}] Total Loss: {avg_loss:.4f} | KD Loss: {avg_kd:.4f} | Teacher Loss: {avg_teacher:.4f}")

        if epoch % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f"kd_diff_epoch{epoch:03d}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved checkpoint at epoch {epoch:03d}")

if __name__ == "__main__":
    main()
