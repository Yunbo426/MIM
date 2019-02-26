# Memory In Memory: A Predictive Neural Network for Learning Higher-Order Non-Stationarity from Spatiotemporal Dynamics

## Abstract

Natural spatiotemporal processes can be highly non-stationary in many ways, e.g. the low-level non-stationarity such as spatial correlations or temporal dependencies of local pixel values; and the high-level non-stationarity such as the accumulation, deformation or dissipation of radar echoes in precipitation forecasting.

We try to stationalize and approximate the non-stationary processes by modeling the differential signals with the MIM recurrent blocks. By stacking multiple MIM blocks, we could potentially handle higher-order non-stationarity. Our model achieves the state-of-the-art results on three spatiotemporal prediction tasks across both synthetic and real-world data.

![model](https://github.com/thuml/MIM/blob/master/readme_fig/readme_structure.png)

## Datasets and Checkpoints:
We uploaded our pre-trained MIM models as well as the training/testing data (Moving MNIST, Color-Changing Moving MNIST, and TaxiBJ) to [here](https://www.dropbox.com/s/7kd82ijezk4lkmp/mim-lib.zip?dl=0). Human3.6M data can be downloaded using data/human36m.sh.

## Generation Results:

#### Moving MNIST:

![mnist1](https://github.com/thuml/MIM/blob/master/readme_fig/mnist1.gif)

![mnist2](https://github.com/thuml/MIM/blob/master/readme_fig/mnist4.gif)

![mnist2](https://github.com/thuml/MIM/blob/master/readme_fig/mnist5.gif)

#### Color-Changing Moving MNIST:

![mnistc1](https://github.com/thuml/MIM/blob/master/readme_fig/mnistc2.gif)

![mnistc2](https://github.com/thuml/MIM/blob/master/readme_fig/mnistc3.gif)

![mnistc2](https://github.com/thuml/MIM/blob/master/readme_fig/mnistc4.gif)

#### Radar Echos:

![radar1](https://github.com/thuml/MIM/blob/master/readme_fig/radar9.gif)

![radar2](https://github.com/thuml/MIM/blob/master/readme_fig/radar3.gif)

![radar3](https://github.com/thuml/MIM/blob/master/readme_fig/radar7.gif)

## BibTeX
```
@article{wang2018memory,
  title={Memory In Memory: A Predictive Neural Network for Learning Higher-Order Non-Stationarity from Spatiotemporal Dynamics},
  author={Wang, Yunbo and Zhang, Jianjin and Zhu, Hongyu and Long, Mingsheng and Wang, Jianmin and Yu, Philip S},
  journal={arXiv preprint arXiv:1811.07490},
  year={2018}
}
```
