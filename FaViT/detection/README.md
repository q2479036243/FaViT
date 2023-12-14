# Applying FaViT to Object Detection

Note: first modify the dataset path in configs/_base_/datasets/*.py

## Evaluation
To evaluate on COCO val2017 on a single node with 8 gpus run:
```
bash dist_test.sh configs/retinanet_favit_b0_fpn_1x_coco.py /path/to/checkpoint_file 8 --out results.pkl --eval bbox
```

## Training
To train on COCO train2017 on a single node with 8 gpus for 12 epochs run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash dist_train.sh configs/retinanet_favit_b0_fpn_1x_coco.py 8 > log/retinanet_favit_b0_fpn_1x_coco.log 2>&1 &
```

## Demo
```
python demo.py demo.jpg /path/to/config_file /path/to/checkpoint_file
```


## Calculating FLOPS & Params

```
python get_flops.py configs/retinanet_favit_b0_fpn_1x_coco.py
```