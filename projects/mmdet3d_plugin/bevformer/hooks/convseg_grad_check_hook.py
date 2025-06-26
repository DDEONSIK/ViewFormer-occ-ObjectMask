# convseg_grad_check_hook.py

import torch
from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class ConvSegGradCheckHook(Hook):
    def after_train_iter(self, runner):
        model = runner.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        try:
            bbox_head = model.pts_bbox_head
            conv_seg = bbox_head.object_mask_module.object_mask_net.decode_head.conv_seg
            # print("[HOOK] AFTER backward() → conv_seg grad 상태")
            # print("  🛠️🚂  conv_seg.weight.requires_grad:", conv_seg.weight.requires_grad)
            # print("  🛠️🚂  conv_seg.weight.grad:", conv_seg.weight.grad)
            # print("  🛠️🚂  conv_seg.bias.requires_grad:", conv_seg.bias.requires_grad)
            # print("  🛠️🚂  conv_seg.bias.grad:", conv_seg.bias.grad)
        except Exception as e:
            print("[HOOK ERROR] conv_seg grad 확인 중 오류 발생:", e)
