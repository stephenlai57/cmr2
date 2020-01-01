# TL;DR

This repo include the port of the following work to Python3 and latest PyTorch [1.3], and remove the cupy/chainer dependancy:

1. [Learning Category-Specific Mesh Reconstruction from Image Collections (ECCV 2018)](https://github.com/akanazawa/cmr)
2. [Neural 3D Mesh Renderer (CVPR 2018)](https://github.com/hiroharu-kato/neural_renderer)

Special thanks to the them and [neural_renderer_pytorch](https://github.com/daniilidis-group/neural_renderer).

![Teaser Image](https://akanazawa.github.io/cmr/resources/images/teaser.png)

### Requirements
- Python 3
- [PyTorch](https://pytorch.org/) tested on version `1.3.0`

### Installation


#### Setup virtualenv
```
conda create -n cmr2
conda activate cmr2
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```


#### install neural_renderer
```
export CUDA_HOME=/path/to/cuda/ 
```

Install specified [CUDA here](https://developer.nvidia.com/cuda-toolkit-archive) same version as PyTorch `python -c 'import torch;print(torch.version.cuda)'`, for example `10.1.243`->`cuda 10.1`. Make sure you set the right `CUDA_HOME` (e.g. `ls $CUDA_HOME/bin/nvcc` works.)
and then 
```
python setup.py install
```

 
### Demo
1. From the `cmr` directory, download the trained model:
```
wget https://people.eecs.berkeley.edu/~kanazawa/cachedir/cmr/model.tar.gz && tar -vzxf model.tar.gz
```
You should see `cmr/cachedir/snapshots/bird_net/`

2. Run the demo:
```
python demo.py --name bird_net --num_train_epoch 500 --img_path misc/demo_data/img1.jpg
python demo.py --name bird_net --num_train_epoch 500 --img_path misc/demo_data/birdie.jpg
```

### Training
Please see [doc/train.md](train.md)

### Citation
If you use this code for your research, please consider citing:
```
@inProceedings{cmrKanazawa18,
  title={Learning Category-Specific Mesh Reconstruction
  from Image Collections},
  author = {Angjoo Kanazawa and
  Shubham Tulsiani
  and Alexei A. Efros
  and Jitendra Malik},
  booktitle={ECCV},
  year={2018}
}

