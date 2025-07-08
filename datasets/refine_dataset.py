import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RefineNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str or Path): 根目录，包含多个 sample_xxxxx 子目录
            transform (callable, optional): 对图像应用的变换（ToTensor 等）
        """
        self.root_dir = Path(root_dir)
        self.sample_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        I0 = Image.open(sample_dir / "I0.png").convert("RGB")
        I1 = Image.open(sample_dir / "I1.png").convert("RGB")
        I_pred = Image.open(sample_dir / "I_pred.png").convert("RGB")
        I_gt = Image.open(sample_dir / "I_gt.png").convert("RGB")

        # 转为 tensor
        I0 = self.transform(I0)
        I1 = self.transform(I1)
        I_pred = self.transform(I_pred)
        I_gt = self.transform(I_gt)

        return I0, I1, I_pred, I_gt
