# Learning to Restore ssTEM Images from Deformation and Corruption

**Accepted by ECCVW-2020**



Wei Huang, Chang Chen, Zhiwei Xiong(*), Yueyi Zhang, Dong Liu, Feng Wu

*Corresponding Author

University of Science and Technology of China (USTC)



## Introduction

This repository is the **official implementation** of the [paper](https://link.springer.com/chapter/10.1007/978-3-030-66415-2_26), "Learning to Restore ssTEM Images from Deformation and Corruption", where more implementation details are presented.



## Installation

This code was tested with Pytorch 0.4.0, CUDA 8.0, Python 3.6.2 and Ubuntu 14.04. It is worth mentioning that, besides some commonly used image processing packages, you also need to install some special post-processing packages for neuron segmentation, such as [waterz](https://github.com/funkey/waterz).

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows,

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/renwu527/auto-emseg:v3.1
# used for interpolation module
docker pull registry.cn-hangzhou.aliyuncs.com/renwu527/auto-emseg:v5.4
# used for unfolding and fusion modules
```

or

```shell
docker pull renwu527/auto-emseg:v3.1 # used for interpolation module
docker pull renwu527/auto-emseg:v5.4 # used for unfolding and fusion modules
```



## Dataset

| Set                   | Size              | Download (Processed)                                         |
| --------------------- | ----------------- | ------------------------------------------------------------ |
| Training set          | (4000x3)x512x512  | [BaiduYun](https://pan.baidu.com/s/1HN9BuyenVtOGprDIDwT0eg) (Access code: weih) |
| Validation set        | (100x3)x1024x1024 | [BaiduYun](https://pan.baidu.com/s/1fTZ95r3etQIglSpAH0ny8A) (Access code: weih) |
| Training set-interp   | 4000x512x512      | [BaiduYun](https://pan.baidu.com/s/1rXeZospmJp3GuUnLzBkjaw) (Access code: weih) |
| Validation set-interp | 100x1024x1024     | [BaiduYun](https://pan.baidu.com/s/1Y0wN0baMVt_LRRjFTMRpMg) (Access code: weih) |
| Training set-SFF      | 4000x512x512      | [BaiduYun](https://pan.baidu.com/s/1nQGKtOAXOkuyRvTDwZ0w6Q) (Access code: weih) |
| Validation set-SFF    | (100x4)x1024x1024 | [BaiduYun](https://pan.baidu.com/s/1Xu09oJtWsIeM0MmgTLD3XA) (Access code: weih) |
| Test set-cremia       | 25x1660x1660      | [BaiduYun](https://pan.baidu.com/s/14X7RTWuuX7U9PgHSZ0m_Gg) (Access code: weih) |
| Test set-cremib       | 25x1660x1660      | [BaiduYun](https://pan.baidu.com/s/1IfuVgjbkEyb183xrzxvKvQ) (Access code: weih) |
| Test set-cremic       | 25x1660x1660      | [BaiduYun](https://pan.baidu.com/s/1GS0aT3M9v8MtICmIM4HuXQ) (Access code: weih) |
| Test set-real         | 10x2048x2048      | [BaiduYun](https://pan.baidu.com/s/1NUU_RbOexR_Q1bgePm8KRg) (Access code: weih) |

Download and unzip them in corresponding folders in './data'.



## Model Zoo

| Module                         | Models                   | Download                                                     |
| ------------------------------ | ------------------------ | ------------------------------------------------------------ |
| Interpolation (L1, default)    | interp.ckpt              | [BaiduYun](https://pan.baidu.com/s/1VXXSBAQYMBQupGjk3k7u0A) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/10yAXgiEcV8dmhDstSpYHhOAkE4rhXknX?usp=sharing) |
| Interpolation (L2)             | interp_l2.ckpt           | [BaiduYun](https://pan.baidu.com/s/1fiysgJk-UJxg-Kv0_g03ow) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/10yAXgiEcV8dmhDstSpYHhOAkE4rhXknX?usp=sharing) |
| Interpolation (ssim)           | interp_ssim.ckpt         | [BaiduYun](https://pan.baidu.com/s/1rLjC_BTxGcs5WTO1WTGIkA) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/10yAXgiEcV8dmhDstSpYHhOAkE4rhXknX?usp=sharing) |
| Interpolation (vgg)            | interp_vgg.ckpt          | [BaiduYun](https://pan.baidu.com/s/12frrDSKVCO22olSgpBOrBw) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/10yAXgiEcV8dmhDstSpYHhOAkE4rhXknX?usp=sharing) |
| Interpolation (adv)            | interp_adv.ckpt          | [BaiduYun](https://pan.baidu.com/s/1wXYEvAVW664C-zPI41esjg) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/10yAXgiEcV8dmhDstSpYHhOAkE4rhXknX?usp=sharing) |
| Unfolding (FusionNet, default) | unfolding_fusionnet.ckpt | [BaiduYun](https://pan.baidu.com/s/1Q4wPpzFbaYIByH2T6OV3tQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/10yAXgiEcV8dmhDstSpYHhOAkE4rhXknX?usp=sharing) |
| Unfolding (FlowNetS)           | unfolding_flownetS.ckpt  | [BaiduYun](https://pan.baidu.com/s/1mpI1tqI97IYGRLSwNLBf1w) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/10yAXgiEcV8dmhDstSpYHhOAkE4rhXknX?usp=sharing) |
| Unfolding (FlowNetC)           | unfolding_flownetC.ckpt  | [BaiduYun](https://pan.baidu.com/s/1J6-h2xxFyrmEBMEjMvarmQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/10yAXgiEcV8dmhDstSpYHhOAkE4rhXknX?usp=sharing) |
| Fusion (UNet, default)         | fusion.ckpt              | [BaiduYun](https://pan.baidu.com/s/1NsirbJ7vP6NChQjJZErtrQ) (Access code: weih) or [GoogleDrive](https://drive.google.com/drive/folders/10yAXgiEcV8dmhDstSpYHhOAkE4rhXknX?usp=sharing) |



## Training and Test

Train on 4 NVIDIA Titan XP GPUs with 32 bathsizes

Test on single GPU with 1 bathsize

### 1. Interpolation module

```she
cd ./scripts_interp
```

#### Install packages

```shell
# install sepconv
cd ./libs/sepconv
./install.bach
cd ../..
# install attrdict
pip install attrdict
# install tensorboardX
pip install tensorboardX
```

#### Training

```shell
python main.py -c=ms_l1loss_decay
```

#### Predict single image

```shell
python python inference_singleImage.py -c=ms_l1loss_decay -id=interp -i1=/PATH/IMAGE1.png -i2=/PATH/IMAGE2.png -o=/PATH/OUTPUT.png
```

#### Predict multiply images

```shell
# For cremia
python inference.py -c=ms_l1loss_decay -id=interp -ip=../data/test/cremia -t=cremia_25sff -op=../results/cremia/
# mean_PSNR=22.7065, mean_SSIM=0.6596

# For cremib
python inference.py -c=ms_l1loss_decay -id=interp -ip=../data/test/cremib -t=cremib_25sff -op=../results/cremib/
# mean_PSNR=22.2252, mean_SSIM=0.6041

# For cremic
python inference.py -c=ms_l1loss_decay -id=interp -ip=../data/test/cremic -t=cremic_25sff -op=../results/cremic/
# mean_PSNR=21.9687, mean_SSIM=0.5767
```

#### Generate interpolation results of training set for subsequent folding and fusion modules

```shell
python inference_trainingset.py
```



### 2. Unfolding module

```shell
cd ./scripts_unfolding
```

#### Training

```shell
python main_flowfusionnet.py -c=sff_flowfusionnet_L1_lr0001decay
```

#### Predict multiply images

```shell
# Middle results for unfolding SFF images.
# Note that it is integrated in the 'inference.py' of fusion module.
# Therefore, you can skip this step if you don't want to obtain the unfolded images.
# For cremia
python inference.py -c=sff_flowfusionnet_L1_lr0001decay -id=unfolding_fusionnet -ip=../data/test/cremia -t=cremia_25sff -op=../results/cremia

# For cremib
python inference.py -c=sff_flowfusionnet_L1_lr0001decay -id=unfolding_fusionnet -ip=../data/test/cremib -t=cremib_25sff -op=../results/cremib

# For cremic
python inference.py -c=sff_flowfusionnet_L1_lr0001decay -id=unfolding_fusionnet -ip=../data/test/cremic -t=cremic_25sff -op=../results/cremic
```



### 3. Fusion module

```shell
cd ./scripts_fusion
```

#### Training

```shell
python main_fusion.py -c=sff_fusion_L1_lr0001decay
```

#### Predict multiply images

```shell
# For cremia
python inference.py -c=sff_fusion_L1_lr0001decay -id=fusion -ip=../data/test/cremia -t=cremia_25sff -op=../results/cremia
# mean_PSNR=26.4418, mean_SSIM=0.8368

# For cremib
python inference.py -c=sff_fusion_L1_lr0001decay -id=fusion -ip=../data/test/cremib -t=cremib_25sff -op=../results/cremib
# mean_PSNR=26.8161, mean_SSIM=0.8202

# For cremic
python inference.py -c=sff_fusion_L1_lr0001decay -id=fusion -ip=../data/test/cremic -t=cremic_25sff -op=../results/cremic
# mean_PSNR=25.7472, mean_SSIM=0.7957
```



## Contact

If you have any problem with the released code, please do not hesitate to contact me by email (weih527@mail.ustc.edu.cn).

