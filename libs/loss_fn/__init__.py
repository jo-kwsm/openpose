import torch.nn as nn

from libs.loss_fn.oploss import OpenPoseLoss

__all__ = ["get_criterion"]

def get_criterion() -> nn.Module:
    criterion = OpenPoseLoss()

    return criterion
