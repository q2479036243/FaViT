import argparse

import torch
from mmcv import Config, DictAction

from mmseg.models import build_segmentor

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')
import favit
from mmcv.cnn.utils.flops_counter import get_model_complexity_info, flops_to_string, params_to_string


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[512, 512],
        help='input image size')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def favit_flops(h, w, r, dim):
    dim1 = dim // 16 * 14
    dim2 = dim // 16 * 2
    flops1 = 2 * h * w * 7 * 7 * dim1
    flops2 = 2 * h * w * (h // r) * (w // r) * dim2
    flops = flops1 + flops2
    return flops


def get_flops(model, input_shape):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False)

    backbone = model.backbone
    backbone_name = type(backbone).__name__

    if 'favit' in backbone_name:
        _, H, W = input_shape
        stage1 = favit_flops(H // 4, W // 4,
                            backbone.block1[0].attn.sr_ratio,
                            backbone.block1[0].attn.dim) * len(backbone.block1)
        stage2 = favit_flops(H // 8, W // 8,
                            backbone.block2[0].attn.sr_ratio,
                            backbone.block2[0].attn.dim) * len(backbone.block2)
        stage3 = favit_flops(H // 16, W // 16,
                            backbone.block3[0].attn.sr_ratio,
                            backbone.block3[0].attn.dim) * len(backbone.block3)
        stage4 = favit_flops(H // 32, W // 32,
                            backbone.block4[0].attn.sr_ratio,
                            backbone.block4[0].attn.dim) * len(backbone.block4)
        flops += stage1 + stage2 + stage3 + stage4
    return flops_to_string(flops), params_to_string(params)


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
                format(model.__class__.__name__))

    flops, params = get_flops(model, input_shape)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()