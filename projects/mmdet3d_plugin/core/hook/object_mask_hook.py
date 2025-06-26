from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class ObjectMaskEpochUpdater(Hook):
    def before_train_epoch(self, runner):
        # DDP 환경에서는 runner.model.module, 단일 GPU에서는 runner.model를 사용
        if hasattr(runner.model, 'module'):
            runner.model.module.object_mask_module.set_epoch(runner.epoch)
        else:
            runner.model.object_mask_module.set_epoch(runner.epoch)
