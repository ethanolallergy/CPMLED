# Enhancing Low-Light Images with Simultaneous Deblurring: A Global-Local Interaction Approach



## LOL-Blur Dataset
(The datasets are hosted on both Google Drive and BaiduPan)
| Dataset | Link | Number | Description|
| :----- | :--: | :----: | :---- | 
| LOL-Blur | [Google Drive](https://drive.google.com/drive/folders/11HcsiHNvM7JUlbuHIniREdQ2peDUhtwX?usp=sharing) / [BaiduPan (key: dz6u)](https://pan.baidu.com/s/1CPphxCKQJa_iJAGD6YACuA) | 12,000 | A total of 170 videos for training and 30 videos for testing, each of which has 60 frames, amounting to 12,000 paired data. (Note that the first and last 30 frames of each video are NOT consecutive, and their darknesses are simulated differently as well.)|


<details close>
<summary>[Unfold] for detailed description of each folder in LOL-Blur dataset:</summary>

<table>
<td>

| LOL-Blur                 | Description             |
| :----------------------- | :---------------------- |
| low_blur                 | low-light blurry images |
| low_blur_noise           | low-light blurry and noisy images |
| low_sharp                | low-light sharp images |
| high_sharp_scaled        | normal-light sharp images with slightly  brightness reduced (simulate soft-light scenes) |
| high_sharp_original      | normal-light sharp images without brightness reduced |
</td>
</table>

<a name="fn1">[1]</a> This method use distorted image as reference. Please refer to the paper for details.<br>
<a name="fn2">[2]</a> Currently, only naive random forest regression is implemented and **does not** support backward.

</details>

## Models
note: The trained models will be opened after the article is accepted.

## Dependencies and Installation

- Pytorch >= 1.7.1
- CUDA >= 10.1
- Other required packages in `requirements.txt`
```
# create new anaconda env
conda create -n CPMLED python=3.9 -y
conda activate CPMLED

# install python dependencies
pip install -r requirements.txt
python basicsr/setup.py develop
```

## Inference Stage
**You can download the model weights:** [github release](https://github.com/ethanolallergy/CPMLED/releases/download/v1.0.0/CPMLED.pth)

Run the following commands:
```
python  inference_cpmled.py -i dataset_path -w model_weight  -o output_dir 
```


## Evaluation

```
# set evaluation metrics of 'psnr', 'ssim', and 'lpips (vgg)'
python scripts/calculate_iqa_pair.py --result_path 'RESULT_ROOT' --gt_path 'GT_ROOT' --metrics psnr ssim lpips
```

## Statement

We are in the process of submitting the paper Enhancing Low-Light Images with Simultaneous Deblurring: A Global-Local Interaction Approach to The Visual Computer.


## Acknowledgements

This code is built on BasicSR, LEDNet.We calculate evaluation metrics using [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch) toolbox. Thanks for their awesome works.
