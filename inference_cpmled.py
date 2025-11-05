import argparse
import cv2
import glob
from tqdm import tqdm
import torch

from utils import img2tensor, tensor2img, imwrite
from basicsr.utils import scandir
from basicsr.archs.CPMLED_arch import CPMLED
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from torchvision.transforms.functional import normalize
import torch.nn.functional as F

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def main():
    """Inference demo for FeMaSR
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='low_blur_noise', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default='CPMLED.pth', help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='/results_test_LOLBlur/enhanced', help='Output folder')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=6000, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.input.endswith('/'):  # solve when path ends with /
        args.input = args.input[:-1]
    if args.output.endswith('/'):  # solve when path ends with /
        args.output = args.output[:-1]
    result_root = f'{args.output}/{os.path.basename(args.input)}'

    down_factor = 8  # check_image_size
    # set up the model
    net = CPMLED(channels=[32, 64, 128, 128], connection=False).to(device)
    net.load_state_dict(torch.load(args.weight)['params'], strict=False)
    net.eval()

    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))
        paths = sorted(list(scandir(args.input, suffix=('jpg', 'png'), recursive=True, full_path=True)))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        pbar.set_description(f'Test {img_name}')

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img_tensor = img2tensor(img / 255., bgr2rgb=True, float32=True)

        normalize(img_tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        img_tensor = img_tensor.unsqueeze(0).to(device)


        with torch.no_grad():
            # check_image_size
            H, W = img_tensor.shape[2:]
            img_t = check_image_size(img_tensor, down_factor)
            output_t = net(img_t)
            if isinstance(output_t, tuple):
                output_t = output_t[0]
            output_t = output_t[:, :, :H, :W]


            output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))

        del output_t
        torch.cuda.empty_cache()

        output = output.astype('uint8')
        # save restored img
        save_restore_path = path.replace(args.input, result_root)
        imwrite(output, save_restore_path)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()
