import os

import torch
import torch.nn as nn

from .openpose import OpenPoseNet

__all__ = ["get_model"]


def get_model(pretrained: bool = True) -> nn.Module:
    model = OpenPoseNet(pretrained=pretrained)

    return model
