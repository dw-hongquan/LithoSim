'''
Author: Hongquan
Date: Apr. 18, 2025
'''
import math
import torch
from torch import Tensor
import sys
sys.path.append('.')

def continuous_positional_encoding_1d(d_model: int, values: Tensor) -> Tensor:
    """
        :param values: [batch_size, 1], range [-1, 1]
        :return: [batch_size, d_model]
    """
    if d_model % 2 != 0:
        raise ValueError("d_model must be divisible by 2")
    div_term = torch.exp(torch.arange(0, d_model, 2, device=values.device).float() * (-math.log(10000.0) / d_model))
    batch_size, _ = values.shape
    pe = torch.zeros(batch_size, d_model, device=values.device)
    scaled_time = values * div_term  # [B, 1] * [1, d_model//2] => [B, d_model//2]
    pe[:, 0::2] = torch.sin(scaled_time)
    pe[:, 1::2] = torch.cos(scaled_time)
    return pe.contiguous()

def continuous_positional_encoding_2d(d_model: int, coords: Tensor) -> Tensor:
    """
        :param coords: [batch_size, N, 2]
        :return: [batch_size, N, d_model]
    """
    if d_model % 4 != 0:
        raise ValueError("d_model must be divisible by 4")
    d_half = d_model // 2
    div_term = torch.exp(torch.arange(0., d_half, 2, device=coords.device).float() * -(math.log(10000.0) / d_half))

    batch_size, N, _ = coords.shape

    x = coords[..., 0]
    y = coords[..., 1]
    pe = torch.zeros(batch_size, N, d_model, device=coords.device)

    # x-pos embed
    x_encoding = x.unsqueeze(-1) * div_term
    pe[..., 0::4] = torch.sin(x_encoding)
    pe[..., 1::4] = torch.cos(x_encoding)

    # y-pos embed
    y_encoding = y.unsqueeze(-1) * div_term
    pe[..., 2::4] = torch.sin(y_encoding)
    pe[..., 3::4] = torch.cos(y_encoding)

    return pe.contiguous()


if __name__ == "__main__":
    pass