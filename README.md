# Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection

This is the Pytorch implementation of our paper - <br>

**Attention-based Class-Conditioned Alignment for Multi-Source Domain Adaptive Object Detection**<br>
Atif Belal, Akhil Meethal, Francisco Perdigon Romero, Marco Pedersoli, Eric Granger

[[arXiv](https://arxiv.org/abs/2403.09918)] 


<p align="center">
<img src="ACIA_model.png" width="85%">
</p>

# Installation

## Prerequisites

- Linux or macOS with Python ≥ 3.8
- PyTorch ≥ 1.5 and torchvision that matches the PyTorch installation.
- Detectron2



## Create Python environment and Install PyTorch

```shell
mkdir python_env
python3 -m venv python_env/
source python_env/bin/activate
pip3 install torch torchvision torchaudio
```

## Build Detectron2 from Source

Follow the [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) to install Detectron2.

## Dataset download

1. Download the datasets

BDD100K - Download the dataset from [BDD100K page](https://bdd-data.berkeley.edu)

Cityscapes - Follow the instruction on [DA-Faster-RCNN](https://github.com/yuhuayc/da-faster-rcnn/tree/master/prepare_data)

Kitty - Download the dataset from [Kitty](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d)

MS COCO - Download the dataset from [COCO](https://cocodataset.org/#download)

Synscapes - Follow the instruction on [Synscapes](https://synscapes.on.liu.se/download.html)


2. Organize the dataset as following:

```shell
ACIA/
└── datasets/
    └── Daytime/
        ├── Annotations/
        ├── Imagesets/
        ├── JPEGImages/

  └── Night/
          ├── Annotations/
          ├── Imagesets/
          ├── JPEGImages/
  └── DuskDawn/
        ├── Annotations/
        ├── Imagesets/
        ├── JPEGImages/
  └── Kitty/
        ├── Annotations/
        ├── Imagesets/
        ├── JPEGImages/
  └── Cityscapes/
        ├── annotations/
        ├── gtFine/
        ├── leftIm8bits
  └── Synscapes/
        ├── img/
        ├── meta/
  └── Coco/
        ├── annotations/
        ├── train2017/
```

## Training

 Train the ACIA under Daytime, Night (source) and DuskDawn (target)

```shell
python train_net.py \
      --num-gpus 4 \
      --config configs/cross_time.yaml \
      OUTPUT_DIR output/cross_time
```

Train the ACIA under Cityscapes, Kitty (source) and Daytime (target)

```shell
python train_net.py \
      --num-gpus 4 \
      --config configs/cross_camera.yaml \
      OUTPUT_DIR output/cross_camera
```

## Resume Training

```shell
python train_net.py \
      --resume \
      --num-gpus 4 \
      --config configs/cross_time.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

## Evaluation

```shell
python train_net.py \
      --eval-only \
      --num-gpus 4 \
      --config configs/crosstime.yaml \
      MODEL.WEIGHTS <your weight>.pth
```

## Pre-trained Models
Crosstime Setting - Coming Soon

Crosscamera Setting - Coming Soon

Mixed Setting - Coming Soon

## Detection Visualization
<p align="center">
<img src="Detection.png" width="95%">
</p>

## Result comparison on the Crosstime and Crosscamera Setting
<p align="center">
<img src="Cross_time_camera.png" width="95%">
</p>

## Result comparison on the Mixed Setting
<p align="center">
<img src="Mixed.png" width="95%">
</p>
