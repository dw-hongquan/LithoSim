'''
Author: Hongquan
Date: Apr. 22, 2025
Description: convolutional FNO (CFNO) for litho.
'''
import sys
sys.path.append('.')

from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange
from src.models.components.litho_embed_net import ChunkedLithoAttn
from src.models.components.basic_net import Downsample, Upsample, Encoder, Decoder


class CFNO(nn.Module): 
    def __init__(self, in_channels: int = 1, out_channels: int = 16, chunk_size: int = 16, num_norm_group: int = 8) -> None:
        super().__init__()
        assert out_channels % num_norm_group == 0, f"out_channels {out_channels} must be divisible by num_norm_group: {num_norm_group}"

        self.chunk_size = chunk_size
        self.fc = nn.Linear(in_channels * (self.chunk_size ** 2), out_channels, dtype = torch.complex64)
        self.conv = nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 1)
        self.norm = nn.GroupNorm(num_norm_group, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        _, C, H, W = x.shape
        h = H // self.chunk_size
        w = W // self.chunk_size
        patches = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1 = self.chunk_size, s2 = self.chunk_size).contiguous()
        patches = patches.view(-1, C * (self.chunk_size ** 2)).contiguous()
        fft = torch.fft.fft(patches, dim = -1, norm = 'ortho')
        fc = self.fc(fft)
        ifft = torch.fft.ifft(fc, norm = 'ortho').real
        ifft = rearrange(ifft, '(b h w) d -> b d h w', h = h, w = w).contiguous()
        return self.norm(self.conv(ifft))

class CFNOFusedLitho(nn.Module):
    def __init__(self, cfno_pool: Tuple = [128, 128], ch_mult: Tuple = [16, 32], z_channels: int = 64, compress_factor: int = 16, source_d_model: int = 8, value_d_model: int = 8, mask_chunk_size: int = 64, source_chunk_size: int = 256) -> None:
        super().__init__()
        self.cfno_pool = cfno_pool
        embed_channels = z_channels + sum(ch_mult)

        self.cfno_block = nn.ModuleList()
        self.mask_down = Downsample(1, True)
        self.resist_up = Upsample(1, True)
        for i_level in ch_mult:
            self.cfno_block.append(CFNO(out_channels = i_level, chunk_size = i_level))

        self.litho_attn = ChunkedLithoAttn(compress_factor, source_d_model, value_d_model, embed_channels, mask_chunk_size, source_chunk_size)

        self.encoder = Encoder(ch = 64,
                               out_ch = 1,
                               ch_mult = [1, 1, 2, 2, 4],
                               num_res_blocks = 1,
                               attn_resolutions = [16],
                               dropout = 0.2,
                               resamp_with_conv = True,
                               in_channels = 1,
                               resolution = 256,
                               z_channels = z_channels,
                               double_z = False)

        self.decoder = Decoder(ch = 64,
                               out_ch = 1,
                               ch_mult = [1, 1, 2, 2, 4],
                               num_res_blocks = 1,
                               attn_resolutions = [16],
                               dropout = 0.2,
                               resamp_with_conv = True,
                               in_channels = 1,
                               resolution = 256,
                               z_channels = embed_channels,
                               double_z = False)

    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        mask_down = self.mask_down(mask)
        mask_embed: Tensor = self.encoder(mask_down)
        for block in self.cfno_block:
            cfno_embed = F.interpolate(block(mask), size = self.cfno_pool)
            mask_embed = torch.cat([mask_embed, cfno_embed], dim = 1)
        mask_attn: Tensor = self.litho_attn(source, mask_embed, dose, defocus)
        resist_embed = self.decoder(mask_attn)
        return self.resist_up(resist_embed)

if __name__ == "__main__":
    pass