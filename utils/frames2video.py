import cv2
import os
import argparse
from typing import List, Tuple


class VideoGenerator:
    """Video Generator from Frames Sequences"""
    # Supported video formats and corresponding codecs
    VIDEO_CODECS = {
        '.avi': 'DIVX',
        '.mp4': 'mp4v',
        '.mov': 'XVID'
    }
    
    def __init__(self, 
                 frame_path: str, 
                 video_path: str, 
                 fps: int = 25,
                 img_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')):
        """Initialization
        Args:
            frame_path (Union[str, Path]): Input directory containing frames.
            video_path (Union[str, Path]): Output directory for generated video.
            img_format (str): Image format for output frames ('png' or 'jpg').
            scale (float): Scale factor for resizing frames.
        """
        self.frame_path = frame_path
        self.video_path = video_path
        self.fps = fps
        self.img_extensions = img_extensions
        
        # Ensure directories exist
        os.makedirs(frame_path, exist_ok=True)
        if os.path.dirname(video_path):
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    def _get_sorted_images(self) -> List[str]:
        """Get and sort image files from directory"""
        images = [img for img in os.listdir(self.frame_path) 
                 if img.lower().endswith(self.img_extensions)]
        images.sort()  # Sort by filename
        return images
    
    def _get_video_codec(self) -> str:
        """Determine video codec based on file extension"""
        ext = os.path.splitext(self.video_path)[1].lower()
        return self.VIDEO_CODECS.get(ext, 'DIVX')  # Default to DIVX
    
    def generate(self) -> bool:
        try:
            images = self._get_sorted_images()
            if not images:
                print(f"Error: No image files found in {self.frame_path}")
                return False
                
            # Get dimensions from first image
            first_image_path = os.path.join(self.frame_path, images[0])
            frame = cv2.imread(first_image_path)
            if frame is None:
                print(f"Error: Failed to read image {first_image_path}")
                return False
            height, width, _ = frame.shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*self._get_video_codec())
            video = cv2.VideoWriter(self.video_path, fourcc, self.fps, (width, height))
            
            # Write all frames
            for idx, image in enumerate(images, 1):
                img_path = os.path.join(self.frame_path, image)
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"Warning: Skipping unreadable image {image}")
                    continue
                
                # Select specific frames to create low fps video if needed
                if idx % 3 == 0:  # Example condition for low fps
                    video.write(frame)

                # video.write(frame)
                print(f"\rProcessing: {idx}/{len(images)}", end='')
            
            video.release()
            print(f"\nVideo successfully saved to {self.video_path}")
            return True
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            return False


def setup_arg_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser"""
    parser = argparse.ArgumentParser(
        description='Generate videos from image sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        '--input_dir',
        default='../Sample/Video2Frames/sample_vid',
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output_file',
        default='../Sample/Frames2Video/combined_video.mp4',
        help='Path for output video file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--fps',
        type=int,
        default=25,
        help='Frames per second for output video'
    )
    parser.add_argument(
        '--extensions',
        default='.png,.jpg,.jpeg',
        help='Comma-separated image file extensions to include'
    )
    parser.add_argument(
        '--hfr',
        action='store_true',
        help='Enable high frame rate mode (doubles the fps)'
    )
    parser.add_argument(
        '--codec',
        choices=['divx', 'mp4v', 'xvid'],
        default='divx',
        help='Video codec to use'
    )
    
    return parser


def main():
    # Parse command line arguments
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Configure based on arguments
    if args.hfr:
        args.fps *= 2
        print(f"High frame rate enabled. Adjusted fps to {args.fps}")
    
    # Create and run video generator
    generator = VideoGenerator(
        frame_path=args.input_dir,
        video_path=args.output_file,
        fps=args.fps,
        img_extensions=tuple(ext.lower() for ext in args.extensions.split(',')))
    
    success = generator.generate()
    
    # Exit with appropriate status code
    exit(0 if success else 1)


if __name__ == "__main__":
    main()