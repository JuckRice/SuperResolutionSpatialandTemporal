import os
import subprocess
import argparse
from pathlib import Path
from typing import List


class VideoProcessor:
    
    def __init__(self, 
                 input_root: str, 
                 output_root: str,
                 checkpoint: str,
                 config: str,
                 model: str = 'adacofnet',
                 gpu_id: int = 0,
                 kernel_size: int = 11,
                 dilation: int = 2):
        """
        Initiate the VideoProcessor with required parameters.
        
        Args:
            input_root: 输入根目录
            output_root: 输出根目录
            checkpoint: 模型检查点路径
            config: 配置文件路径
            model: 模型名称
            gpu_id: GPU ID
            kernel_size: 内核大小
            dilation: 膨胀系数
        """
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.checkpoint = Path(checkpoint)
        self.config = Path(config)
        self.model = model
        self.gpu_id = gpu_id
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.output_root.mkdir(parents=True, exist_ok=True)
        
    def find_image_folders(self) -> List[Path]:
        """Search for folders containing images in the input directory."""
        folders = []
        for item in self.input_root.iterdir():
            if item.is_dir():
                if any(file.suffix.lower() in ('.png', '.jpg', '.jpeg') 
                       for file in item.iterdir()):
                    folders.append(item)
        return folders
    
    def run_interpolate(self, folder: Path):
        """call interpolate_video.py script"""
        output_dir = self.output_root / folder.name
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            'python', 
            'interpolate_video.py',
            '--input_video', str(folder),
            '--output_video', str(output_dir),
            '--checkpoint', str(self.checkpoint),
            '--config', str(self.config),
            '--model', self.model,
            '--gpu_id', str(self.gpu_id),
            '--kernel_size', str(self.kernel_size),
            '--dilation', str(self.dilation)
        ]
        
        print(f"\nProcessing folder: {folder.name}")
        print(f"\nKernal size: {self.kernel_size}, Checkpoint: {self.checkpoint}")
        print("Running command:", ' '.join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, cmd)
        return output_dir
    
    def run_frame2video(self, output_dir: Path):
        """调用帧合成视频脚本"""
        video_path = self.output_root / f"{output_dir.name}.mp4"
        
        cmd = [
            'python',
            'frame2video.py',
            str(output_dir),
            str(video_path),
            '--fps', '50'  # 假设插值后帧率提高
        ]
        
        print(f"\nGenerating video: {video_path.name}")
        subprocess.run(cmd, check=True)
        return video_path
    
    def process_all(self):
        """处理所有找到的图像文件夹"""
        folders = self.find_image_folders()
        print(f"Found {len(folders)} folders to process")
        
        for folder in folders:
            try:
                print(f"\n{'='*40}")
                print(f"Starting processing: {folder.name}")
                
                # 第一步：运行插值处理
                output_dir = self.run_interpolate(folder)
                
                # 第二步：生成视频
                video_path = self.run_frame2video(output_dir)
                
                print(f"Successfully created {video_path}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error processing {folder.name}: {str(e)}")
            except Exception as e:
                print(f"Unexpected error with {folder.name}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='完全适配您工作流程的视频处理自动化脚本'
    )
    
    # 必需参数
    parser.add_argument('input_root', help='输入根目录')
    parser.add_argument('output_root', help='输出根目录')
    
    # 设为可选参数并提供默认值
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--model', type=str, default='adacofnet')
    
    parser.add_argument('--checkpoint', type=str, default='./checkpoint/kernelsize_5/ckpt.pth')
    parser.add_argument('--config', type=str, default='./checkpoint/kernelsize_5/config.txt')
    parser.add_argument('--kernel_size', type=int, default=5)

    parser.add_argument('--dilation', type=int, default=1)

    parser.add_argument('--index_from', type=int, default=0, help='when index starts from 1 or 0 or else')
    parser.add_argument('--zpad', type=int, default=3, help='zero padding of frame name.')

    parser.add_argument('--input_video', type=str, default='./sample_video')
    parser.add_argument('--output_video', type=str, default='./interpolated_video')
    
    args = parser.parse_args()
    
    processor = VideoProcessor(
        input_root=args.input_root,
        output_root=args.output_root,
        checkpoint=args.checkpoint,
        config=args.config,
        model=args.model,
        gpu_id=args.gpu_id,
        kernel_size=args.kernel_size,
        dilation=args.dilation
    )
    processor.process_all()
    
    print("\nAll processing completed!")


if __name__ == "__main__":
    main()