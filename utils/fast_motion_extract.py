import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path


def estimate_flow(img1, img2):
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(img1, img2, None)  # H x W x 2
    flow = torch.from_numpy(flow).permute(2, 0, 1)  # 2 x H x W
    return flow


def process_one_sequence(seq_path, save_root, global_count, flow_thresh=8.0):
    frame_paths = sorted(Path(seq_path).glob("*.png"))
    saved = 0

    for i in range(1, len(frame_paths) - 1):
        I0 = cv2.imread(str(frame_paths[i - 1]))
        I1 = cv2.imread(str(frame_paths[i + 1]))
        I_gt = cv2.imread(str(frame_paths[i]))

        gray0 = cv2.cvtColor(I0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)

        flow = estimate_flow(gray0, gray1)  # 2xHxW
        mag = torch.sqrt(flow[0] ** 2 + flow[1] ** 2)
        mean_mag = mag.mean().item()

        if mean_mag > flow_thresh:
            triplet_dir = Path(save_root) / f"sample_{global_count:05d}"
            triplet_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(triplet_dir / "I0.png"), I0)
            cv2.imwrite(str(triplet_dir / "I1.png"), I1)
            cv2.imwrite(str(triplet_dir / "I_gt.png"), I_gt)
            global_count += 1
            saved += 1

    return global_count, saved


def extract_from_all_sequences(dataset_root, save_root, flow_thresh=8.0):
    global_count = 0
    dataset_root = Path(dataset_root)
    subdirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    for subdir in tqdm(subdirs, desc="Processing folders"):
        print(f"\nProcessing folder: {subdir.name}")
        global_count, saved = process_one_sequence(
            seq_path=subdir,
            save_root=save_root,
            global_count=global_count,
            flow_thresh=flow_thresh
        )
        print(f"Processed {subdir.name}: kept {saved} samples")

    print(f"\n总共保存了 {global_count} 个高速运动样本到 {save_root}")


if __name__ == "__main__":
    # 假设每个文件夹是一段视频帧序列
    extract_from_all_sequences(
        dataset_root="../Dataset/STVT_dataset/",     # 输入帧文件夹
        save_root="../fast_motion_triplets/",          # 输出保存路径
        flow_thresh=8.0                               # 光流幅度阈值（可调整）
    )
