import sys
from pathlib import Path

ljh_dir = str(Path(__file__).parent)
sys.path.append(ljh_dir)

import torch
import torch.nn as nn


class Loss(nn.Module):
    """Loss function based on the paper"""
