
## Real-time Controllable Denoising for Image and Video

#### Zhaoyang Zhang, Yitong Jiang, Wenqi Shao, Xiaogang Wang, Ping Luo, Kaimo Lin, Jinwei Gu

The official pytorch implementation of the paper **[Real-time Controllable Denoising for Image and Video]**

| [Github](https://github.com/jiangyitong/RCD) |  [Page](https://zzyfd.github.io/RCD-page/) | [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Real-Time_Controllable_Denoising_for_Image_and_Video_CVPR_2023_paper.pdf) |  [Arxiv](https://arxiv.org/abs/2303.16425) | 


## Demo Video
<video src="https://github.com/zzyfd/RCD-page/assets/13939478/0f75950f-bb72-45f0-9a80-f882de7a5c50" controls="controls" width="1000">
</video>



## Installation
This implementation based on [BasicSR] 

Basic requirements:
```python
python 3.9.12
pytorch 1.12.1
cuda 11.8
```
Other requirements:
```
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```
## Quick Start 
Data Preparation:
  1. Download [Nam dataset](https://shnnam.github.io/research/ccnoise/)
  2. Crop the gt and input images into 512*512 patches and save as gt.lmdb and input.lmdb, respectively. (or download from Google Drive [GT](https://drive.google.com/file/d/1Cyi5ZCjBPHixa8zE5YuUnjXvm9LBeQLI/view?usp=share_link) and [Input](https://drive.google.com/file/d/1aGmgGJupzNiseAOVD6CUoZOtg-kB3Usz/view?usp=sharing))
  3. Edit the dataroot_lq and dataroot_gt in NAFNet-RCD-tiny.yml to the corresponding paths: /your_path/gt.lmdb and /your_path/input.lmdb

Test Nam real image noise dataset with NAFNet-RCD-tiny model, which is trained on SIDD training dataset

```
python basicsr/test.py --opt options/test/NAFNet-RCD-tiny.yml
```

Result structure:

  Groundtruth

      imageName_gt.png 

  Denoise levels

      imageName_level_0.png imageName_level_1.png imageName_level_2.png imageName_level_3.png imageName_level_4.png 

  AutoTune results

      imageName_res.png

## Comments
Our codebase is based on the [NAFNet](https://github.com/megvii-research/NAFNet)

## Citation
If you find our paper useful for your research, please consider citing our work :blush: : 
```
@InProceedings{Zhang_2023_CVPR,
    author    = {Zhang, Zhaoyang and Jiang, Yitong and Shao, Wenqi and Wang, Xiaogang and Luo, Ping and Lin, Kaimo and Gu, Jinwei},
    title     = {Real-Time Controllable Denoising for Image and Video},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {14028-14038}
}
```

