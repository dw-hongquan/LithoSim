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


# 示例使用
if __name__ == "__main__":
    # coords = torch.randn(2, 7180, 2)
    # pos_emb = continuous_positional_encoding_2d(d_model=32, coords=coords)

    # # 与 value 结合（假设 value 形状为 [2, 7180, 1]）
    # values = torch.rand(2, 7180, 1)
    # features = torch.cat([pos_emb, values], dim=-1)  # [2, 7180, 65]
    # print(features.shape)
    # 参数设置
    B = 4
    d_model = 6

    # 示例输入（已归一化到[-1, 1]）
    x = torch.tensor([[-0.1], [0.3], [0.6], [-0.5]])  # 形状 [4, 1]

    # 初始化编码器
    pe = continuous_positional_encoding_1d(d_model, x)

    print("输入数据形状:", x.shape)
    print("位置编码形状:", pe.shape)
    print("\n具体编码示例（保留4位小数）:")
    print(pe.detach().numpy().round(4))