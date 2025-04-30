import cv2
import os
import argparse
import json
from pathlib import Path
from typing import Union
from typing import List, Tuple

class VideoSplitter:
    """Video Frame Extractor"""
    
    def __init__(self,
                 input_path: Union[str, Path],
                 output_root: Union[str, Path],
                 img_format: str = 'png',
                 scale: float = 1.0):
        """Initialization
        Args:
            input_path (Union[str, Path]): Input video file or directory containing videos.
            output_root (Union[str, Path]): Output directory for extracted frames.
            img_format (str): Image format for output frames ('png' or 'jpg').
            scale (float): Scale factor for resizing frames.
        """
        self.input_path = Path(input_path)
        self.output_root = Path(output_root)
        self.img_format = img_format.lower()
        self.scale = scale
        self.supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
        
        # Make sure the output directory exists
        self.output_root.mkdir(parents=True, exist_ok=True)
        
    def _get_video_files(self) -> List[Path]:
        """Get video files from input path"""
        if self.input_path.is_dir():
            return [f for f in self.input_path.iterdir() 
                    if f.suffix.lower() in self.supported_formats]
        return [self.input_path] if self.input_path.exists() else []
    
    def _get_metadata(self, video_path: Path) -> dict:
        """Get video metadata"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        return {
            'fps': fps,
            'frame_count': frame_count,
            'original_resolution': (width, height),
            'filename': video_path.name
        }
    
    def _process_single_video(self, video_path: Path):
        """Process a single video file to extract frames"""
        # Get metadata
        metadata = self._get_metadata(video_path)
        
        # Create output directory for this video
        output_dir = self.output_root / video_path.stem
        output_dir.mkdir(exist_ok=True)
        
        # save metadata to JSON
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Begin frame extraction
        print(f"Begin processing {video_path.name}，Total frames: {metadata['frame_count']}")
        cap = cv2.VideoCapture(str(video_path))
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Adjust frame size if scale is not 1.0
            if self.scale != 1.0:
                frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale)
                
            # Save the frame
            img_name = f"frame_{frame_idx:06d}.{self.img_format}"
            cv2.imwrite(str(output_dir / img_name), frame)
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"\rProcessing {video_path.name}: Extracted {frame_idx}/{metadata['frame_count']} frames", end='')
        
        cap.release()
        print(f"\nFinish {video_path.name} extraction，{frame_idx} frames saved.")
        
    def process_all(self):
        """Process all videos in the input path"""
        video_files = self._get_video_files()
        print(f"{len(video_files)} video files found.")
        
        for video_path in video_files:
            try:
                print(f"\n{'='*40}")
                print(f"Start processing: {video_path.name}")
                self._process_single_video(video_path)
            except Exception as e:
                print(f"Error occured when processinng {video_path.name}: {str(e)}")
                continue

def main():
    parser = argparse.ArgumentParser(description='Video to Frames Converter',)
    # Required arguments
    parser.add_argument(
        '--input_dir',
        default='../Sample/sample_vid.mp4',
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output_file',
        default='../Sample/Video2Frames',
        help='Path for output video file'
    )
    parser.add_argument(
        '--img_format',
        type=str,
        default='png',
        choices=['png', 'jpg'],
        help='output format(default: png)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='scale (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    splitter = VideoSplitter(
        input_path=args.input_dir,
        output_root=args.output_file,
        img_format=args.img_format,
        scale=args.scale
    )
    splitter.process_all()
    print("\nAll process done！")

if __name__ == "__main__":
    main()