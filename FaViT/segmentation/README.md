# Applying FaViT to Semantic Segmentation

## Evaluation
To evaluate on a single node with 8 gpus run:
```
bash dist_test.sh configs/sem_fpn/FaViT/fpn_favit_b0_ade20k_160k.py /path/to/checkpoint_file 8 --out results.pkl --eval mIoU
```

## Training
To train on a single node with 8 gpus run:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup bash dist_train.sh configs/sem_fpn/FaViT/fpn_favit_b0_ade20k_160k.py 8 > log/fpn_favit_b0_ade20k_160k.log 2>&1 &
```

## Calculating FLOPS & Params

```
python get_flops.py configs/sem_fpn/FaViT/fpn_favit_b0_ade20k_160k.py
```

