# A Neural Galerkin Solver for Accurate Surface Reconstruction

### [**Paper**](https://dl.acm.org/doi/abs/10.1145/3550454.3555457) | [**Video**](https://youtu.be/QT5k0ZxDFfo) | [**Talk**](https://youtu.be/dYy-lzeMsCQ)

![](./assets/teaser.jpg)

This repository contains the implementation of the above paper. It is accepted to **ACM SIGGRAPH Asia 2022**.
- Authors: [Jiahui Huang](https://cg.cs.tsinghua.edu.cn/people/~huangjh/), [Hao-Xiang Chen](), [Shi-Min Hu](https://cg.cs.tsinghua.edu.cn/shimin.htm)
    - Contact Jiahui either via email or github issues.


If you find our code or paper useful, please consider citing:
```bibtex
@article{huang2022neuralgalerkin,
  author = {Huang, Jiahui and Chen, Hao-Xiang and Hu, Shi-Min},
  title = {A Neural Galerkin Solver for Accurate Surface Reconstruction},
  year = {2022},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {41},
  number = {6},
  doi = {10.1145/3550454.3555457},
  journal = {ACM Trans. Graph.},
}
```

## Introduction

NeuralGalerkin is a method to reconstruct triangular meshes from point clouds. 
Please note that this implementation is accelerated by [Jittor]() deep learning framework. The framework is based on Just-in-Time compilation of the code, and is maintained by a great team led by Prof. Shi-Min Hu. 
To fully unlock the training and inference code used to reproduce our results, please checkout the `pth` branch.

## Getting started

We suggest to use [Anaconda](https://www.anaconda.com/) to manage your environment. Following is the suggested way to install the dependencies:

```bash
# Create a new conda environment
conda create -n ngs python=3.10
conda activate ngs

# Install other packages
pip install -r requirements.txt

# Compile pytorch_spsr CUDA extensions inplace
python setup.py build_ext --inplace
```

## Experiments

Please follow the commands below to run all of our experiments.
As our framework is accelerated by Jittor, we only provide inference interfaces here. Please checkout `pth` branch for the full loops.

### ShapeNet

Please download the dataset from [here](https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/data/dataset_small_v1.1.zip), and put the extracted `onet` folder under `data/shapenet`.

- 1K input, No noise (trained model download [here](https://drive.google.com/file/d/1WMYrhTtvTCWRVbMZiVfxmB3fvvNASUwe/view?usp=sharing))
```shell
# Test our trained model (add -v to visualize)
python test.py none --ckpt checkpoints/shapenet-perfect1k/main/paper/checkpoints/best.ckpt 
```

- 3K input, Small noise (trained model download [here](https://drive.google.com/file/d/1aE7XAnl8ffdbU22F6ZZkhiU9-zoNGAt-/view?usp=sharing))
```shell
# Test our trained model (add -v to visualize)
python test.py none --ckpt checkpoints/shapenet-noise3k/main/paper/checkpoints/best.ckpt 
```

- 3K input, Large noise (trained model download [here](https://drive.google.com/file/d/13CGwy3k4Mny6__zbDHrFiiZaUkv1CelT/view?usp=sharing))
```shell
# Test our trained model (add -v to visualize)
python test.py none --ckpt checkpoints/shapenet-noiser3k/main/paper/checkpoints/best.ckpt 
```

### Matterport3D

Please download the dataset from [here](https://drive.google.com/file/d/18c02XjpWHtP7vjFhQyuokH90G8ikxo23/view?usp=sharing), and put the extracted `matterport` folder under `data/`.

- Without Normal (trained model download [here](https://drive.google.com/file/d/1ouek3Ywt8QVf-9D55KhF_C_15SSvRg8Y/view?usp=sharing))
```shell
# Test our trained model (add -v to visualize)
python test.py none --ckpt checkpoints/matterport/without_normal/paper/checkpoints/best.ckpt
```

- With Normal (trained model download [here](https://drive.google.com/file/d/1ouek3Ywt8QVf-9D55KhF_C_15SSvRg8Y/view?usp=sharing))
```shell
# Test our trained model (add -v to visualize)
python test.py none --ckpt checkpoints/matterport/with_normal/paper/checkpoints/best.ckpt 
```

### D-FAUST

Please download the dataset from [here](https://drive.google.com/file/d/1ghL6RRQjZAEj4jCfozW11pDWypPnpp7m/view?usp=sharing), and put the extracted `dfaust` folder under `data/`.

- Origin split (trained models download [here](https://drive.google.com/file/d/1zjpMhymlAbYQv2jplYw4eWfkywVzpoKd/view?usp=sharing))
```shell
# Test our trained model (add -v to visualize)
python test.py none --ckpt checkpoints/dfaust/origin/paper/checkpoints/best.ckpt 
```

- Novel split (test only)
```shell
# Test our trained model (add -v to visualize)
python test.py configs/dfaust/data_10k_novel.yaml --ckpt checkpoints/dfaust/origin/paper/checkpoints/best.ckpt -v
```

## Acknowledgements

We thank anonymous reviewers for their constructive feedback. 
This work was supported by the National Key R&D Program of China (No. 2021ZD0112902), Research Grant of Beijing Higher Institution Engineering Research Center and Tsinghua-Tencent Joint Laboratory for Internet Innovation Technology.

Part of the code is directly borrowed from [torchsparse](https://github.com/mit-han-lab/torchsparse) and [Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks).
