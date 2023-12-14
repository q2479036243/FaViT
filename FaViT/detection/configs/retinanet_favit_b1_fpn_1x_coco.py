_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/coco_detection.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    pretrained='pretrained/favit_b1.pth',
    backbone=dict(
        type='favit_b1',
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
