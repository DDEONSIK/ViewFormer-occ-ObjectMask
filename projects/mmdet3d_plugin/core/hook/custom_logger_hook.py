from mmcv.runner import HOOKS, LoggerHook

@HOOKS.register_module()
class CustomLoggerHook(LoggerHook):
    def log(self, runner):
        # runner.outputs는 loss dict.
        log_vars = runner.outputs
        # 예: loss_object_mask가 존재하면 출력하도록 함
        if 'loss_object_mask' in log_vars:
            self.logger.info(f"loss_object_mask: {log_vars['loss_object_mask']:.4f}")
        # 기본 텍스트 로그 출력 방식도 유지
        super().log(runner)
