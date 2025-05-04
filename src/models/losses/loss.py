'''
Author: Hongquan
Date: Apr. 15, 2025
'''

from torch import nn, Tensor
import torch.nn.functional as F

class LithoLoss(nn.Module):
    def __init__(self, dice_weight: int = 1, bce_weight: int = 1):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: Tensor, target: Tensor, smooth: float = 1.0e-5) -> Tensor:
        pred_data = F.sigmoid(pred)
        bce_loss = F.binary_cross_entropy(pred_data, target)
        bs = target.shape[0]
        pred_flatten = pred_data.view(bs, -1).contiguous()
        target_flatten = target.view(bs, -1).contiguous()
        intersection = pred_flatten * target_flatten
        dice = (2. * intersection.sum(1) + smooth) / (pred_flatten.sum(1) + target_flatten.sum(1) + smooth)
        dice_loss = 1 - dice.sum() / bs

        return dice_loss * self.dice_weight + bce_loss * self.bce_weight