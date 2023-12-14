_base_ = [
    '../../_base_/models/fpn_r50.py',
    '../../_base_/datasets/ade20k.py',
    '../../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/favit_b3.pth',
    backbone=dict(
        type='favit_b3',
        style='pytorch'),
    neck=dict(in_channels=[96, 192, 384, 768]),
    decode_head=dict(num_classes=150))


gpu_multiples = 2
# optimizer
optimizer = dict(type='AdamW', lr=0.00006*gpu_multiples, weight_decay=0.01)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=160000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=16000//gpu_multiples)
evaluation = dict(interval=16000//gpu_multiples, metric='mIoU')


