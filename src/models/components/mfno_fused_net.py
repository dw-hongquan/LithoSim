'''
Author: Hongquan
Date: Apr. 22, 2025
Description: mixed FNO (MFNO) for litho.
'''
import sys
sys.path.append('.')

import torch
from torch import nn, Tensor
from torch.nn import init
from einops import rearrange
from typing import Tuple
from src.models.components.litho_embed_net import ChunkedLithoAttn
from src.models.components.basic_net import Downsample, Upsample, Encoder, Decoder

class MFNO(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 16,
                 chunk_size: int = 64, modes1: int = 16, modes2: int = 16):
        super().__init__()
        assert chunk_size % modes1 == 0 and chunk_size % modes2 == 0, f"chunk_size {chunk_size} must be divisible by modes: ({modes1}, {modes2})"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.chunk_size = chunk_size
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (in_channels * modes1 * modes2) ** -0.5
        self.weights_ll = nn.Parameter(
            self.scale * torch.randn(out_channels, in_channels, modes1, modes2, dtype=torch.complex64)
        )
        self.weights_lh = nn.Parameter(
            self.scale * torch.randn(out_channels, in_channels, modes1, modes2, dtype=torch.complex64)
        )
        self.weights_hl = nn.Parameter(
            self.scale * torch.randn(out_channels, in_channels, modes1, modes2, dtype=torch.complex64)
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

        self.post_norm = nn.LayerNorm(out_channels)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert H % self.chunk_size == 0 and W % self.chunk_size == 0, f"Input dimensions ({H}, {W}) must be divisible by chunk_size: {self.chunk_size}"

        h_dim, w_dim = H // self.chunk_size, W // self.chunk_size

        x_chunks = rearrange(x, 'b c (h s1) (w s2) -> b (h w) c s1 s2', s1 = self.chunk_size, s2 = self.chunk_size).contiguous()
        batch_total = B * h_dim * w_dim

        fft_x = torch.fft.fft2(x_chunks.reshape(batch_total, C, self.chunk_size, self.chunk_size).contiguous(), norm = 'ortho')

        out_fft = torch.zeros(batch_total, self.out_channels, self.chunk_size, self.chunk_size, dtype = torch.complex64, device = x.device)
        out_fft[:, :, :self.modes1, :self.modes2] = torch.einsum('bcij,ocij->boij', fft_x[:, :, :self.modes1, :self.modes2].contiguous(), self.weights_ll) * self.scale
        out_fft[:, :, :self.modes1, -self.modes2:] = torch.einsum('bcij,ocij->boij', fft_x[:, :, -self.modes1:, :self.modes2].contiguous(), self.weights_lh) * self.scale
        out_fft[:, :, -self.modes1:, :self.modes2] = torch.einsum('bcij,ocij->boij', fft_x[:, :, -self.modes1:, :self.modes2].contiguous(), self.weights_hl) * self.scale

        x_trans = torch.fft.ifft2(out_fft.contiguous(), norm = 'ortho').real
        x_trans = self.post_norm(x_trans.permute(0,2,3,1).contiguous()).permute(0,3,1,2).contiguous()
        x_recon = rearrange(x_trans.view(B, h_dim * w_dim, self.out_channels, self.chunk_size, self.chunk_size).contiguous(), 'b (h w) c s1 s2 -> b c (h s1) (w s2)', h=h_dim, w=w_dim).contiguous()

        return self.conv_block(x_recon)


class MFNOFusedLitho(nn.Module):
    def __init__(self, mask_channels: int = 8, ch_mult: Tuple = [16, 32], z_channels: int = 64, compress_factor: int = 16, source_d_model: int = 8, value_d_model: int = 8, mask_chunk_size: int = 64, source_chunk_size: int = 256) -> None:
        super().__init__()
        self.mfno_block = nn.ModuleList()
        self.mask_conv = nn.Conv2d(1, mask_channels, 1)
        embed_channels = mask_channels + sum(ch_mult)
        self.mask_down = Downsample(embed_channels, True)
        self.resist_up = Upsample(1, True)
        for i_level in ch_mult:
            self.mfno_block.append(MFNO(out_channels = i_level, chunk_size = i_level, modes1 = i_level // 2, modes2 = i_level // 2))

        self.litho_attn = ChunkedLithoAttn(compress_factor, source_d_model, value_d_model, z_channels, mask_chunk_size, source_chunk_size)

        self.encoder = Encoder(ch = 64,
                               out_ch = 1,
                               ch_mult = [1, 1, 2, 2, 4],
                               num_res_blocks = 1,
                               attn_resolutions = [16],
                               dropout = 0.2,
                               resamp_with_conv = True,
                               in_channels = embed_channels,
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
                               z_channels = z_channels,
                               double_z = False)

    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        mask_feature: Tensor = self.mask_conv(mask)
        for block in self.mfno_block:
            mfno_embed: Tensor = block(mask)
            mask_feature = torch.cat([mask_feature, mfno_embed], dim = 1)
        mask_down: Tensor = self.mask_down(mask_feature)
        mask_embed = self.encoder(mask_down)
        mask_attn: Tensor = self.litho_attn(source, mask_embed, dose, defocus)
        resist_embed = self.decoder(mask_attn)
        return self.resist_up(resist_embed)

if __name__ == '__main__':
    pass