import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.refinenet import RefinementNet
from losses.refineloss import charbonnier_loss, PerceptualLoss, EdgeLoss, CombinedLoss
from datasets.refine_dataset import RefineNetDataset

parser = argparse.ArgumentParser(description="Train RefineNet with configurable losses")

parser.add_argument('--use_perceptual', action='store_true', help="Include Perceptual loss")
parser.add_argument('--use_edge', action='store_true', help="Include Edge loss")

parser.add_argument('--w_c', type=float, default=1.0, help="Weight for Charbonnier loss")
parser.add_argument('--w_p', type=float, default=0.1, help="Weight for Perceptual loss")
parser.add_argument('--w_e', type=float, default=0.05, help="Weight for Edge loss")

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=8)

parser.add_argument('--save_dir', type=str, default='./checkpoints/RefineNet_Default_Path')
parser.add_argument('--data_root', type=str, default='../fast_motion_triplets')

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)  # Ensure directory exists

    train_dataset = RefineNetDataset(root_dir=args.data_root)  # (I0, I1, AdaCoF_output, GT)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = RefinementNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Initialize loss function and weights of CombinedLoss
    perceptual = PerceptualLoss().to(device) if args.use_perceptual else None
    edge = EdgeLoss().to(device) if args.use_edge else None
    loss_fn = CombinedLoss(
        w_c=1.0,
        w_p=0.1,
        w_e=0.05
    ).to(device)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_l_c = 0.0
        total_l_p = 0.0
        total_l_e = 0.0
        start_time = time.time()  # record start time for this epoch

        for I0, I1, I_pred, I_gt in train_loader:
            I0, I1, I_pred, I_gt = I0.to(device), I1.to(device), I_pred.to(device), I_gt.to(device)

            optimizer.zero_grad()
            output = model(I0, I1, I_pred)

            # Compute losses
            l_char = charbonnier_loss(output, I_gt)
            l_perc = perceptual(output, I_gt) if perceptual else 0.0
            l_edge = edge(output, I_gt) if edge else 0.0

            loss = args.w_c * l_char
            if perceptual: loss += args.w_p * l_perc
            if edge: loss += args.w_e * l_edge

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_l_c += l_char.item()
            total_l_p += l_perc.item() if perceptual else 0.0
            total_l_e += l_edge.item() if edge else 0.0

            avg_loss = total_loss / len(train_loader)
            avg_char = total_l_c / len(train_loader)
            avg_perc = total_l_p / len(train_loader)
            avg_edge = total_l_e / len(train_loader)

            end_time = time.time() # record end time for this epoch

        print(f"[Epoch {epoch:03d}] Total: {avg_loss:.4f} | Charb: {avg_char:.4f}"
              + (f" | Perc: {avg_perc:.4f}" if perceptual else "")
              + (f" | Edge: {avg_edge:.4f}" if edge else "")
              + (f" | Time: {end_time - start_time:.2f}s" if args.epochs >= 1 else ""))

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            ckpt_path = os.path.join(args.save_dir, f"refine_epoch{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()