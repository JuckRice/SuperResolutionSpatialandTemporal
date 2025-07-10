import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.refinenet import RefinementNet
from losses.refineloss import charbonnier_loss, CombinedLoss
from datasets.refine_dataset import RefineNetDataset

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = RefineNetDataset(root_dir="../fast_motion_triplets")  # (I0, I1, AdaCoF_output, GT)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize model, loss function, and optimizer
model = RefinementNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialize loss function and weightS of CombinedLoss
loss_fn = CombinedLoss(
    w_c=1.0,
    w_p=0.1,
    w_e=0.05
).to(device)

save_dir = "./checkpoints/RefineNet_Charb"  # RefineNet checkpoints
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

# Training loop
num_epochs = 5
for epoch in range(1, num_epochs + 1):
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

        # Calculate losses
        l_char = charbonnier_loss(output, I_gt)
        l_perc = loss_fn.perceptual(output, I_gt)
        l_edge = loss_fn.edge(output, I_gt)
        loss = loss_fn.w_c * l_char + loss_fn.w_p * l_perc + loss_fn.w_e * l_edge

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_l_c += l_char.item()
        total_l_p += l_perc.item()
        total_l_e += l_edge.item()

        avg_loss = total_loss / len(train_loader)
        avg_l_c = total_l_c / len(train_loader)
        avg_l_p = total_l_p / len(train_loader)
        avg_l_e = total_l_e / len(train_loader)

        end_time = time.time() # record end time for this epoch

    print(f"[Epoch {epoch:03d}] Total: {avg_loss:.4f} | Charb: {avg_l_c:.4f} | Perc: {avg_l_p:.4f} | Edge: {avg_l_e:.4f}\
        Time: {end_time - start_time:.2f}s")

    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        ckpt_path = os.path.join(save_dir, f"refine_epoch{epoch:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"✅ Saved checkpoint to {ckpt_path}")
