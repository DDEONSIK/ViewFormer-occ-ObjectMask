# /home/hyun/local_storage/code/vieeew/ViewFormer-Occ/tools/analysis_tools/get_flops.py
# Copyright (c) OpenMMLab. All rights reserved.

import argparse
import torch
from mmcv import Config, DictAction

from mmdet3d.models import build_model

# try:
#     from mmcv.cnn import get_model_complexity_info
# except ImportError:
#     raise ImportError('Please upgrade mmcv to >0.6.2')

from thop import profile


# 커스텀 모델 및 데이터셋 임포트 추가
from projects.mmdet3d_plugin.models.detectors.viewformer import ViewFormer
from projects.mmdet3d_plugin.datasets.nuscenes_occ import NuSceneOcc

def parse_args():
    parser = argparse.ArgumentParser(description='Calculate FLOPs and Params of a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file path')  # 수정: checkpoint 인자 추가
    parser.add_argument('--shape', type=int, nargs='+', default=[40000, 4], help='input point cloud size')
    parser.add_argument('--modality', type=str, default='point', choices=['point', 'image', 'multi'], help='input data modality')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings in the used config, the key-value pair '
                                                                           'in xxx=yyy format will be merged into config file. If the value to '
                                                                           'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                                                                           'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                                                                           'Note that the quotation marks are necessary and that no white space '
                                                                           'is allowed.')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 모달리티에 따른 입력 형태 설정
    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)
    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3, ) + tuple(args.shape)
        elif len(args.shape) == 4:
            input_shape = tuple(args.shape)  # [num_cams, 3, H, W]
        else:
            raise ValueError('invalid input shape')
    elif args.modality == 'multi':
        raise NotImplementedError('FLOPs counter is currently not supported for models with multi-modality input')

    # 모델 빌드
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()



    # forward_dummy로 대체 (다중 프레임 입력 지원)
    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError('FLOPs counter is currently not supported for {}'.format(model.__class__.__name__))

    # 다중 프레임 입력 설정
    dummy_input = torch.randn(1, 6, 3, 224, 224).cuda() # (batch_size, temporal_frames, num_cams, channels, height, width)

    # FLOPs 및 파라미터 수 계산 (전체 레이어별 상세 출력)
    flops, params = profile(model, inputs=(dummy_input,), verbose=True)
    print(model)  # 모델 구조 출력
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {dummy_input.shape}\nFLOPs: {flops / 1e9:.2f} GFLOPs\nParams: {params / 1e6:.2f} M\n{split_line}')


if __name__ == '__main__':
    main()
