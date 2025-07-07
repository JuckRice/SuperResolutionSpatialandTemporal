import argparse
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Batch_triplets_process')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='adacofnet')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--config', type=str, default='./checkpoint/kernelsize_5/config.txt')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

parser.add_argument('--triplet_root', type=str, default='../fast_motion_triplets')
parser.add_argument('--interpolate_script', type=str, default='./interpolate_twoframe.py')


def main():
    args = parser.parse_args()
    triplet_dirs = sorted([d for d in Path(args.triplet_root).iterdir() if d.is_dir()])

    for triplet_dir in tqdm(triplet_dirs, desc="Processing triplets"):
        path_I0 = triplet_dir / "I0.png"
        path_I1 = triplet_dir / "I1.png"
        path_out = triplet_dir / "I_pred.png"

        if not path_I0.exists() or not path_I1.exists():
            print(f"Skipping {triplet_dir}: missing input images")
            continue

        cmd = [
            "python", args.interpolate_script,
            "--gpu_id", str(args.gpu_id),
            "--model", args.model,
            "--checkpoint", args.checkpoint,
            "--config", args.config,
            "--kernel_size", str(args.kernel_size),
            "--dilation", str(args.dilation),
            "--first_frame", str(path_I0),
            "--second_frame", str(path_I1),
            "--output_frame", str(path_out)
        ]

        subprocess.run(cmd)

    print("All triplets processed, results are I_pred.png.")

if __name__ == "__main__":
    main()