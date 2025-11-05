import torch
import os
import cv2
import argparse
import os.path as osp
import numpy as np
import glob
from basicsr.utils import scandir
import pyiqa

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='[path_to_results]')
    parser.add_argument('--gt_path', type=str, default='[path_to_gt]')
    parser.add_argument('--metrics', nargs='+', default=['psnr','ssim','lpips'])
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces')

    args = parser.parse_args()

    if args.result_path.endswith('/'):  # solve when path ends with /
        args.result_path = args.result_path[:-1]
    if args.gt_path.endswith('/'):  # solve when path ends with /
        args.gt_path = args.gt_path[:-1]

    # Initialize metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iqa_psnr, iqa_ssim, iqa_lpips = None, None, None
    score_psnr_all, score_ssim_all, score_lpips_all = [], [], []
    print(args.metrics)
    if 'psnr' in args.metrics:
      iqa_psnr = pyiqa.create_metric('psnr').to(device)
      iqa_psnr.eval()
    if 'ssim' in args.metrics:
      iqa_ssim = pyiqa.create_metric('ssim').to(device)
      iqa_ssim.eval()
    if 'lpips' in args.metrics:
      # iqa_lpips = pyiqa.create_metric('lpips').to(device)
      iqa_lpips = pyiqa.create_metric('lpips-vgg').to(device)
      iqa_lpips.eval() 

    img_out_paths = sorted(list(scandir(args.result_path, suffix=('jpg', 'png'), 
                                    recursive=True, full_path=True)))
    total_num = len(img_out_paths)

    for i, img_out_path in enumerate(img_out_paths):
        img_name = os.path.basename(img_out_path)  # 只获取文件名
        cur_i = i + 1
        print(f'[{cur_i}/{total_num}] Processing: {img_name}')
        img_out = cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        if img_out is None:
            print(f'Error reading output image: {img_out_path}')
            continue
        img_out = np.transpose(img_out, (2, 0, 1))
        img_out = torch.from_numpy(img_out).float()
    # for i, img_out_path in enumerate(img_out_paths):
    #     img_name = img_out_path.replace(args.result_path+'/', '')
    #     cur_i = i + 1
    #     print(f'[{cur_i}/{total_num}] Processing: {img_name}')
    #     img_out = cv2.imread(img_out_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/255.
    #     img_out = np.transpose(img_out, (2, 0, 1))
    #     img_out = torch.from_numpy(img_out).float()
        try:
            # 关键修复：直接拼接GT路径（忽略result_path的子目录结构）
            # img_name = "0089.png" 这样的纯文件名
            subdir = osp.basename(osp.dirname(img_out_path))  # 获取"0256"
            img_gt_path = osp.join(args.gt_path, subdir, img_name)

            if not osp.exists(img_gt_path):
                raise FileNotFoundError(f"GT file not found: {img_gt_path}")

            img_gt = cv2.imread(img_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            if img_gt is None:
                raise ValueError(f"Failed to read GT image: {img_gt_path}")

            img_gt = np.transpose(img_gt, (2, 0, 1))
            img_gt = torch.from_numpy(img_gt).float()

            with torch.no_grad():
                img_out = img_out.unsqueeze(0).to(device)
                img_gt = img_gt.unsqueeze(0).to(device)
                if iqa_psnr is not None:
                    score_psnr_all.append(iqa_psnr(img_out, img_gt).item())
                if iqa_ssim is not None:
                    score_ssim_all.append(iqa_ssim(img_out, img_gt).item())
                if iqa_lpips is not None:
                    score_lpips_all.append(iqa_lpips(img_out, img_gt).item())
          # img_gt_path = img_out_path.replace(args.result_path, args.gt_path)
          # img_gt = cv2.imread(img_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)/255.
          # img_gt = np.transpose(img_gt, (2, 0, 1))
          # img_gt = torch.from_numpy(img_gt).float()
          # with torch.no_grad():
          #   img_out = img_out.unsqueeze(0).to(device)
          #   img_gt = img_gt.unsqueeze(0).to(device)
          #   if iqa_psnr is not None:
          #     score_psnr_all.append(iqa_psnr(img_out, img_gt).item())
          #   if iqa_ssim is not None:
          #     score_ssim_all.append(iqa_ssim(img_out, img_gt).item())
          #   if iqa_lpips is not None:
          #     score_lpips_all.append(iqa_lpips(img_out, img_gt).item())
        except Exception as e:
            print(f'skip: {img_name} - Reason: {str(e)}')
            continue
        if (i+1)%20 == 0:
          print(f'[{cur_i}/{total_num}] PSNR: {sum(score_psnr_all)/len(score_psnr_all)}, \
                  SSIM: {sum(score_ssim_all)/len(score_ssim_all)}, \
                  LPIPS: {sum(score_lpips_all)/len(score_lpips_all)}\n')

    print('-------------------Final Scores-------------------\n')
    if len(score_psnr_all) == 0:
        print("ERROR: No valid images processed!")
    else:
        psnr_avg = sum(score_psnr_all) / len(score_psnr_all)
        ssim_avg = sum(score_ssim_all) / len(score_ssim_all) if score_ssim_all else 0
        lpips_avg = sum(score_lpips_all) / len(score_lpips_all) if score_lpips_all else 0
        print(f'PSNR: {psnr_avg}, SSIM: {ssim_avg}, LPIPS: {lpips_avg}')
        # print(f'PSNR: {sum(score_psnr_all)/len(score_psnr_all)}, \
        #         SSIM: {sum(score_ssim_all)/len(score_ssim_all)}, \
        #         LPIPS: {sum(score_lpips_all)/len(score_lpips_all)}')

