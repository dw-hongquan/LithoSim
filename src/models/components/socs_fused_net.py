'''
Author: Hongquan
Date: Apr. 22, 2025
Description: Sum of Coherent Sources Approach (SOCS) for litho.
'''

import sys
sys.path.append('.')

import torch
from torch import Tensor, nn, complex64
from src.models.components.litho_embed_net import ComplexChunkedLithoAttn
from src.models.components.complex_net import ComplexConvUpsample, ComplexConvDownsample, ComplexEncoder, ComplexDecoder

class SOCSFusedLitho(nn.Module):
    def __init__(self, scale_num: int = 3, z_channels: int = 64, compress_factor: int = 16, source_d_model: int = 8, value_d_model: int = 8, mask_chunk_size: int = 16, source_chunk_size: int = 256) -> None:
        super().__init__()
        self.down_block = nn.ModuleList()
        self.up_block = nn.ModuleList()
        for _ in range(scale_num):
            self.down_block.append(ComplexConvDownsample(1, True))
            self.up_block.append(ComplexConvUpsample(1, True))

        self.to_resist = nn.Conv2d(1, 1, 1)
        self.litho_attn = ComplexChunkedLithoAttn(compress_factor, source_d_model, value_d_model, z_channels, mask_chunk_size, source_chunk_size)

        self.encoder = ComplexEncoder(ch = 64,
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

        self.decoder = ComplexDecoder(ch = 64,
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

    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        mask_spat: Tensor = torch.fft.ifftshift(mask).to(complex64)
        mask_fft: Tensor = torch.fft.fft2(mask_spat)
        mask_fft = torch.fft.fftshift(mask_fft)
        for down in self.down_block:
            mask_fft = down(mask_fft)

        mask_embed: Tensor = self.encoder(mask_fft)
        mask_attn: Tensor = self.litho_attn(source, mask_embed, dose, defocus)
        aerial_embed: Tensor = self.decoder(mask_attn)
        for up in self.up_block:
            aerial_embed = up(aerial_embed)
        
        aerial_ifft = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(aerial_embed)))
        aerial = torch.sum(torch.abs(aerial_ifft * torch.conj(aerial_ifft)), dim = 1, keepdim=True)

        return self.to_resist(aerial)

if __name__ == '__main__':
    pass
