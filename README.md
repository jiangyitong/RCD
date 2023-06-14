
## Real-time Controllable Denoising for Image and Video

The official pytorch implementation of the paper **[Real-time Controllable Denoising for Image and Video]**


### Installation
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
### Quick Start 
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

### Comments
Our codebase is based on the [NAFNet](https://github.com/megvii-research/NAFNet)



