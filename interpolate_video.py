import argparse
from PIL import Image
import torch
from torchvision import transforms
import models
import os
from torchvision.utils import save_image
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Video Interpolation')

parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--model', type=str, default='adacofnet')
parser.add_argument('--checkpoint', type=str, default='./checkpoint/kernelsize_5/ckpt.pth')
parser.add_argument('--config', type=str, default='./checkpoint/kernelsize_5/config.txt')

parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--dilation', type=int, default=1)

parser.add_argument('--index_from', type=int, default=0, help='when index starts from 1 or 0 or else')
parser.add_argument('--zpad', type=int, default=4, help='zero padding of frame name.')

parser.add_argument('--input_video', type=str, default='./sample_video')
parser.add_argument('--output_video', type=str, default='./interpolated_video')

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main():
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    config_file = open(args.config, 'r')
    while True:
        line = config_file.readline()
        if not line:
            break
        if line.find(':') == 0:
            continue
        else:
            tmp_list = line.split(': ')
            if tmp_list[0] == 'kernel_size':
                args.kernel_size = int(tmp_list[1])
            if tmp_list[0] == 'flow_num':
                args.flow_num = int(tmp_list[1])
            if tmp_list[0] == 'dilation':
                args.dilation = int(tmp_list[1])
    config_file.close()

    model = models.Model(args)

    print('Loading the model...')

    checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'), weights_only=True)
    model.load(checkpoint['state_dict'])

    base_dir = args.input_video

    # check if output video directory exists
    if not os.path.exists(args.output_video):
        os.makedirs(args.output_video)

    # Get all image files in the input directory and sort them
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    frame_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    frame_files = sorted(frame_files)  # Sort files alphabetically

    frame_len = len(frame_files)

    for idx in range(frame_len - 1):
        
        print(idx, '/', frame_len - 1, end='\r')

        frame_name1 = os.path.join(base_dir, frame_files[idx])
        frame_name2 = os.path.join(base_dir, frame_files[idx + 1])

        frame1 = to_variable(transform(Image.open(frame_name1)).unsqueeze(0))
        frame2 = to_variable(transform(Image.open(frame_name2)).unsqueeze(0))

        model.eval()
        frame_out = model(frame1, frame2)

        # interpolate
        frame1_normalized = frame1.clone()
        # !Warning!: Normaliztaion might cause the interpolated frame to be dimmer than the original frames.
        # frame1_normalized = (frame1_normalized - frame1_normalized.min()) / (frame1_normalized.max() - frame1_normalized.min())
        # save_image(frame1_normalized, args.output_video + '/' + str((idx - args.index_from) * 10 + args.index_from).zfill(args.zpad) + '.png')
        save_image(frame1_normalized, os.path.join(args.output_video, f"frame_{((idx - args.index_from) * 10 + args.index_from):04d}.png"))
        
        frame_out_normalized = frame_out.clone()
        # frame_out_normalized = (frame_out_normalized - frame_out_normalized.min()) / (frame_out_normalized.max() - frame_out_normalized.min())
        # save_image(frame_out_normalized, args.output_video + '/' + str((idx - args.index_from) * 10 + 5 + args.index_from).zfill(args.zpad) + '.png')
        save_image(frame_out_normalized, os.path.join(args.output_video, f"frame_{((idx - args.index_from) * 10 + 5 + args.index_from):04d}.png"))

    # last frame
    print(frame_len - 1, '/', frame_len - 1)
    frame_name_last = os.path.join(base_dir, frame_files[-1])
    frame_last = to_variable(transform(Image.open(frame_name_last)).unsqueeze(0))
    frame_last_normalized = frame_last.clone()
    # frame_last_normalized = (frame_last_normalized - frame_last_normalized.min()) / (frame_last_normalized.max() - frame_last_normalized.min())
    save_image(frame_last_normalized, os.path.join(args.output_video, f"frame_{(frame_len - 1) * 10:04d}.png"))
    # save_image(frame_last_normalized, args.output_video + '/' + str((idx - args.index_from) * 10 + args.index_from).zfill(args.zpad) + '.png')



if __name__ == "__main__":
    main()
