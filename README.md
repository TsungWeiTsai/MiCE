## MiCE: Mixture of Contrastive Experts for Unsupervised Image Clustering

This repo includes the PyTorch implementation of the [MiCE paper](https://openreview.net/pdf?id=gV3wdEOGy_V), which is a unified probabilistic clustering framework that simultaneously exploits the discriminative representations
learned by contrastive learning and the semantic structures captured by a latent mixture model. 

### Requirements and Installation
We recommended the following dependencies.

* Python 3.7
* [PyTorch](http://pytorch.org/) (>=1.3.1)
* [sciki-learn](https://scikit-learn.org/)
* [tensorboard-logger](https://github.com/TeamHG-Memex/tensorboard_logger)


### Unsupervised Clustering
To train MiCE on CIFAR-10 with a single GPU, please use the following command line:

```
CUDA_VISIBLE_DEVICES=0 python train_MiCE.py \
  --learning_rate 1.0 --lr_decay_epochs 480,640,800 --lr_decay_rate 0.1 \  
  --model resnet34_cifar --epoch 1000  --dataset cifar10 \
  --tau 1.0 --batch_size 256  
```

For STL-10, the command line looks like:

```
CUDA_VISIBLE_DEVICES=0 python train_MiCE.py \
  --learning_rate 1.0 --lr_decay_epochs 1500,2000,2500 --lr_decay_rate 0.1 \  
  --model resnet34 --epoch 3000  --dataset stl10 \
  --tau 1.0 --batch_size 256  
```


### Evaluation 

With the trained model, we can evaluate the clustering performance of the method on the same dataset using:
```
CUDA_VISIBLE_DEVICES=0 python eval_MiCE.py \ 
    --model resnet34_cifar  --dataset cifar10 --nu 16384 \ 
    --test_path [path to the trained model]
```

#### Acknowledgments
Part of this code is inspired by [CMC](https://github.com/HobbitLong/CMC)
