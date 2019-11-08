# Memory In Memory Networks

MIM is a neural network for video prediction and spatiotemporal modeling. It is based on the paper [Memory In Memory: A Predictive Neural Network for Learning Higher-Order Non-Stationarity from Spatiotemporal Dynamics](https://arxiv.org/pdf/1811.07490.pdf) to be presented at CVPR 2019.

## Abstract

Natural spatiotemporal processes can be highly non-stationary in many ways, e.g. the low-level non-stationarity such as spatial correlations or temporal dependencies of local pixel values; and the high-level non-stationarity such as the accumulation, deformation or dissipation of radar echoes in precipitation forecasting.

We try to stationalize and approximate the non-stationary processes by modeling the differential signals with the MIM recurrent blocks. By stacking multiple MIM blocks, we could potentially handle higher-order non-stationarity. Our model achieves the state-of-the-art results on three spatiotemporal prediction tasks across both synthetic and real-world data.

![model](https://github.com/ZJianjin/mim_images/blob/master/readme_structure.png)

## Pre-trained Models and Datasets

All pre-trained MIM models have been uploaded to [DROPBOX](https://www.dropbox.com/s/7kd82ijezk4lkmp/mim-lib.zip?dl=0) and [BAIDU YUN](https://pan.baidu.com/s/1O07H7l1NTWmAkx3UCDVMLA) (password: srhv).

It also includes our pre-processed training/testing data for Moving MNIST, Color-Changing Moving MNIST, and TaxiBJ. 

For Human3.6M, you may  download it using data/human36m.sh.

## Generation Results

#### Moving MNIST

![mnist1](https://github.com/ZJianjin/mim_images/blob/master/mnist1.gif)

![mnist2](https://github.com/ZJianjin/mim_images/blob/master/mnist4.gif)

![mnist2](https://github.com/ZJianjin/mim_images/blob/master/mnist5.gif)

#### Color-Changing Moving MNIST

![mnistc1](https://github.com/ZJianjin/mim_images/blob/master/mnistc2.gif)

![mnistc2](https://github.com/ZJianjin/mim_images/blob/master/mnistc3.gif)

![mnistc2](https://github.com/ZJianjin/mim_images/blob/master/mnistc4.gif)

#### Radar Echos

![radar1](https://github.com/ZJianjin/mim_images/blob/master/radar9.gif)

![radar2](https://github.com/ZJianjin/mim_images/blob/master/radar3.gif)

![radar3](https://github.com/ZJianjin/mim_images/blob/master/radar7.gif)

#### Human3.6M

![human1](https://github.com/ZJianjin/mim_images/blob/master/human3.gif)

![human2](https://github.com/ZJianjin/mim_images/blob/master/human5.gif)

![human3](https://github.com/ZJianjin/mim_images/blob/master/human10.gif)

## BibTeX
```
@article{wang2018memory,
  title={Memory In Memory: A Predictive Neural Network for Learning Higher-Order Non-Stationarity from Spatiotemporal Dynamics},
  author={Wang, Yunbo and Zhang, Jianjin and Zhu, Hongyu and Long, Mingsheng and Wang, Jianmin and Yu, Philip S},
  journal={arXiv preprint arXiv:1811.07490},
  year={2019}
}
```
