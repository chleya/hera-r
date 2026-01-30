"""
生长实验平台核心模块
"""

from .growing_net import SimpleGrowingNet
from .trainer import GrowthTrainer

__all__ = ["SimpleGrowingNet", "GrowthTrainer"]