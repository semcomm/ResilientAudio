from torch.optim.lr_scheduler import _LRScheduler
import warnings

class WarmupPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(self, optimizer, *, warmup_steps=None, warmup_ratio=None, max_steps=None, min_lr=0.0, last_epoch=-1):
        assert not (
            warmup_steps is not None and warmup_ratio is not None
        ), "Either use particular number of step or ratio"
        assert warmup_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        if step <= self.warmup_steps and self.warmup_steps > 0:
            # print('step: ', step, ' lr:', self._get_warmup_lr(step)) #TODO
            return self._get_warmup_lr(step)

        if (self.max_steps is not None) and (step > self.max_steps):
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_warmup_lr(self, step):
        lr_val = (step + 1) / (self.warmup_steps + 1)
        return [initial_lr * lr_val for initial_lr in self.base_lrs]

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class SquareRootConstantPolicy(_LRScheduler):
    """Adds warmup kwargs and warmup logic to lr policy.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self, optimizer, *, constant_steps=None, constant_ratio=None, max_steps=None, min_lr=0.0, last_epoch=-1
    ):
        assert not (
            constant_steps is not None and constant_ratio is not None
        ), "Either use particular number of step or ratio"
        assert constant_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        # It is necessary to assign all attributes *before* __init__,
        # as class is wrapped by an inner class.
        self.max_steps = max_steps
        if constant_steps is not None:
            self.constant_steps = constant_steps
        elif constant_ratio is not None:
            self.constant_steps = int(constant_ratio * max_steps)
        else:
            self.constant_steps = 0

        self.constant_lr = 1 / (constant_steps ** 0.5)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        if step <= self.constant_steps:
            return [self.constant_lr for _ in self.base_lrs]

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)

    def _get_lr(self, step):
        """Simple const lr policy"""
        return self.base_lrs


class WarmupHoldPolicy(WarmupPolicy):
    """Variant of WarmupPolicy which maintains high learning rate for a defined number of steps.
    All arguments should be passed as kwargs for clarity,
    Args:
        warmup_steps: Number of training steps in warmup stage
        warmup_ratio: Ratio of warmup steps to total steps
        hold_steps: Number of training steps to hold the learning rate after warm up
        hold_ratio: Ratio of hold steps to total steps
        max_steps: Total number of steps while training or `None` for
            infinite training
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps=None,
        warmup_ratio=None,
        hold_steps=None,
        hold_ratio=None,
        max_steps=None,
        min_lr=0.0,
        last_epoch=-1,
    ):
        assert not (hold_steps is not None and hold_ratio is not None), "Either use particular number of step or ratio"
        assert hold_ratio is None or max_steps is not None, "If there is a ratio, there should be a total steps"

        self.min_lr = min_lr
        self._last_warmup_lr = 0.0

        # Necessary to duplicate as class attributes are hidden in inner class
        self.max_steps = max_steps
        if warmup_steps is not None:
            self.warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * max_steps)
        else:
            self.warmup_steps = 0

        if hold_steps is not None:
            self.hold_steps = hold_steps + self.warmup_steps
        elif hold_ratio is not None:
            self.hold_steps = int(hold_ratio * max_steps) + self.warmup_steps
        else:
            self.hold_steps = 0

        super().__init__(
            optimizer,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            max_steps=max_steps,
            last_epoch=last_epoch,
            min_lr=min_lr,
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.", UserWarning
            )

        step = self.last_epoch

        # Warmup phase
        if step <= self.warmup_steps and self.warmup_steps > 0:
            return self._get_warmup_lr(step)

        # Hold phase
        if (step >= self.warmup_steps) and (step < self.hold_steps):
            return self.base_lrs

        if step > self.max_steps:
            return [self.min_lr for _ in self.base_lrs]

        return self._get_lr(step)


def _noam_hold_annealing(initial_lr, step, warmup_steps, hold_steps, decay_rate, min_lr):
    # hold_steps = total number of steps to hold the LR, not the warmup + hold steps.
    T_warmup_decay = max(1, warmup_steps ** decay_rate)
    T_hold_decay = max(1, (step - hold_steps) ** decay_rate)
    lr = (initial_lr * T_warmup_decay) / T_hold_decay
    lr = max(lr, min_lr)
    return lr


class NoamHoldAnnealing(WarmupHoldPolicy):
    def __init__(self, optimizer, *, max_steps, decay_rate=0.5, min_lr=0.0, last_epoch=-1, **kwargs):
        """
        Implementation of the Noam Hold Annealing policy from the SqueezeFormer paper.

        Unlike NoamAnnealing, the peak learning rate can be explicitly set for this scheduler.
        The trainer first performs linear warmup, then holds the peak LR, then decays with some trainer for
        the remainder of the steps. Therefore the min-lr is still dependent on the hyper parameters selected.

        It's trainer is determined by three factors-

        Warmup Steps: Initial stage, where linear warmup occurs uptil the peak LR is reached. Unlike NoamAnnealing,
            the peak LR is explicitly stated here instead of a scaling factor.

        Hold Steps: Intermediate stage, where the peak LR is maintained for some number of steps. In this region,
            the high peak LR allows the model to converge faster if training is stable. However the high LR
            may also cause instability during training. Should usually be a significant fraction of training
            steps (around 30-40% of the entire training steps).

        Decay Steps: Final stage, where the LR rapidly decays with some scaling rate (set by decay rate).
            To attain Noam decay, use 0.5, for Squeezeformer recommended decay, use 1.0. The fast decay after
            prolonged high LR during hold phase allows for rapid convergence.

        References:
            - [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](https://arxiv.org/abs/2206.00888)

        Args:
            optimizer: Pytorch compatible Optimizer object.
            warmup_steps: Number of training steps in warmup stage
            warmup_ratio: Ratio of warmup steps to total steps
            hold_steps: Number of training steps to hold the learning rate after warm up
            hold_ratio: Ratio of hold steps to total steps
            max_steps: Total number of steps while training or `None` for
                infinite training
            decay_rate: Float value describing the polynomial decay after the hold period. Default value
                of 0.5 corresponds to Noam decay.
            min_lr: Minimum learning rate.
        """
        self.decay_rate = decay_rate
        super().__init__(optimizer=optimizer, max_steps=max_steps, last_epoch=last_epoch, min_lr=min_lr, **kwargs)

    def _get_lr(self, step):
        if self.warmup_steps is None or self.warmup_steps == 0:
            raise ValueError("Noam scheduler cannot be used without warmup steps")

        if self.hold_steps > 0:
            hold_steps = self.hold_steps - self.warmup_steps
        else:
            hold_steps = 0

        new_lrs = [
            _noam_hold_annealing(
                initial_lr,
                step=step,
                warmup_steps=self.warmup_steps,
                hold_steps=hold_steps,
                decay_rate=self.decay_rate,
                min_lr=self.min_lr,
            )
            for initial_lr in self.base_lrs
        ]
        return new_lrs


lr_config = {
    "warmup_steps": 20,
    "hold_steps": 160,
    "max_steps": None,
    "decay_rate": 1.0,
    "base_lr": 1e-3,
    "min_lr": 5e-5,
    "weight_decay": 5e-4
}