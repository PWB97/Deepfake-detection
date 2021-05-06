# Learning a Deep Dual-level Network for Robust DeepFake Detection

Wenbo Pu, Jing Hu, Xin Wang, Yuezun Li, Bin Zhu, Bin Kong, Youbing Yin, Qi Song,   Xi Wu, Siwei Lyu

Chengdu University of Information Technology, Chengdu, China; Keya Medical, Seattle, USA; Ocean University of China; Microsoft Research Asia, Beijing, China; University at Buffalo, State University of New York, USA.



## Overview

![](./overview.png)

## Imbalanced Performance

<img src="./imbalanced performance.png" alt="60" style="zoom:50%;" />

## Requirements

- Pytorch 1.4.0
- Ubuntu 16.04
- CUDA 10.0
- Python 3.6

## Usage

- To train a model, use 

  ```shell
  python train.py
  ```

- To test a model, use

  ```shell
  python test.py
  ```

We provided our method, Xception, FWA, Mesonet, Capsule and others to train and test in this repository. Go to **config.py** to change the configurations.

## Training data preparation

We provided a script to generate training and test data for this repository. Use **make_train_test.py**.

## Notice

This repository is NOT for commecial use. It is provided "as it is" and we are not responsible for any subsequence of using this code.
