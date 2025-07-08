import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.refinenet import RefinementNet
from losses.refineloss import charbonnier_loss
from datasets.refine_dataset import RefineNetDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = YourDataset(...)  # 包含 (I0, I1, AdaCoF_output, GT)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 2. 初始化模型
model = RefinementNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 3. 训练循环
for epoch in range(1, 51):
    model.train()
    total_loss = 0.0
    for I0, I1, I_pred, I_gt in train_loader:
        I0, I1, I_pred, I_gt = I0.to(device), I1.to(device), I_pred.to(device), I_gt.to(device)

        optimizer.zero_grad()
        output = model(I0, I1, I_pred)
        loss = charbonnier_loss(output, I_gt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")
