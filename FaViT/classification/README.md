# Factorization Transformer

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Evaluation
To evaluate on ImageNet val with a single GPU run:
```
bash dist_train.sh configs/favit/favit_b0.py 1 --data-path /data3/QHL/DATA/ImageNet2012/ --data-set IMNET --resume /path/to/checkpoint_file --eval
```

## Training
To train on ImageNet or CIFAR100 on a single node with 8 gpus for 300 epochs run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash dist_train.sh configs/favit/favit_b0.py 8 --data-path /data3/QHL/DATA/ImageNet2012/ --data-set IMNET > log/favit_b0_imagenet.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash dist_train.sh configs/favit/favit_b0.py 8 --data-path /data3/publicData/cifar100/ --data-set CIFAR > log/favit_b0_cifar.log 2>&1 &
```

## Calculating FLOPS & Params

```
python get_flops.py favit_b0
```