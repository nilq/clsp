"""Custom schedulers."""
import numpy as np

from typing import Any, Callable

def assign_learning_rate(optimizer: Any, new_lr: float) -> None:
    """Assign learning rate to optimizer.
    
    Args:
        optimizer (Any): Optimizer to assign learning rate to.
        new_lr (float): New learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr: float, warmup_len: int, step: int) -> float:
    """Get warm-up learning rate.

    Args:
        base_lr (float): Base learning rate.
        warmup_len (int): Length of warm-up in steps.
        step (int): Current step.

    Returns:
        float: Warm-up learning rate.
    """
    return base_lr * (step + 1) / warmup_len


def cosine_lr(optimizer: Any, base_lr: float, warmup_len: int, steps: int) -> Callable:
    """Cosine learning rate scheduler.

    Args:
        optimizer (Any): Relevant optimizer.
        base_lr (float): Base learning rate for scheduler.
        warmup_len (int): Warm-up length in steps.
        steps (int): Steps in schedule.

    Returns:
        Callable: Learning rate adjuster.
    """
    def lr_adjuster(step: int) -> float:
        """Get learning rate for step.

        Args:
            step (int): Current step.

        Returns:
            float: Adjusted learning rate for cosine schedule.
        """        

        if step < warmup_len:
            lr = _warmup_lr(base_lr, warmup_len, step)
        else:
            e = step - warmup_len
            es = steps - warmup_len
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr

        assign_learning_rate(optimizer, lr)
        return lr

    return lr_adjuster