# 3DGAN For Designers

<!-- [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/tf-3dgan/blob/master/LICENSE)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1610.07584-brightgreen.svg)](https://arxiv.org/abs/1610.07584)
 -->

## Introuction

* This is a very simple-to-use and simple-to-understand pytorch implementation of part of the [paper](https://arxiv.org/abs/1610.07584) "Learning a Probabilistic Latent Space of Object Shapes via 3D Generative-Adversarial Modeling".

Note that for training, your batch_size has to be at least 2

The code is heavily based on [Simple 3D-GAN-PyTorch](https://github.com/xchhuang/simple-pytorch-3dgan),[3DGAN-Pytorch](https://github.com/rimchang/3DGAN-Pytorch) and [tf-3dgan](https://github.com/meetshah1995/tf-3dgan) and thanks for them.

The major pipeline would be the same as Simple 3D-GAN-PyTorch, except for a couple of major modifications:

* The voxelization process no longer needs MatLab to run due to its limited accessibility (for more detailed expalanation please refer to http://en.people.cn/n3/2020/0612/c90000-9700230.html). The open-source voxelization software binvox is used instead.

* The Training process was tested with 64 × 64 × 64 voxel size, but also allows for flexibility of 32 × 32 × 32 and 128 × 128 × 128 voxel size.

* The visualization tool has also been updated to run interactively in CoLab

### Prerequisites

* Python 3.7.9 | Anaconda4.x
* Pytorch 1.6.0
* tensorboardX 2.1
* matplotlib 2.1
* visdom (optional)