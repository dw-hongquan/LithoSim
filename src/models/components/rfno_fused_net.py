'''
Author: Hongquan
Date: Apr. 22, 2025
Description: Reduced FNO (RFNO) for litho.
'''

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from src.models.components.litho_embed_net import ChunkedLithoAttn
from src.models.components.basic_net import Encoder, Decoder, Downsample, Upsample

class RFNO(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 64, modes1: int = 32, modes2: int = 32, num_norm_group: int = 8) -> None:
        super(RFNO, self).__init__()
        assert out_channels % num_norm_group == 0, f'out_channels {out_channels} must be devided by num_norm_group: {num_norm_group}'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (in_channels * out_channels) ** -0.5
        self.weights0 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, 1, 1, dtype=torch.complex64))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))

        self.norm = nn.GroupNorm(num_norm_group, out_channels)

    def compl_mul2d(self, input: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("bixy,ioxy->boxy", input.contiguous(), weights.contiguous()) * self.scale

    def forward(self, x: Tensor) -> Tensor:
        batch_size, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x.contiguous(), norm = 'ortho')
        x_ft = x_ft * self.weights0

        out_ft = torch.zeros(batch_size, self.out_channels,  H, W // 2, dtype = torch.complex64, device = x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        out = torch.fft.irfft2(out_ft.contiguous(), s = (H, W), norm = 'ortho')
        return self.norm(out.contiguous())

class RFNOFusedLitho(nn.Module):
    def __init__(self, rfno_pool: int = 16, z_channels: int = 64, mode: int = 32, compress_factor: int = 16, source_d_model: int = 8, value_d_model: int = 8, mask_chunk_size: int = 64, source_chunk_size: int = 256) -> None:
        super().__init__()
        self.rfno_pool = rfno_pool
        self.rfno = RFNO(out_channels = z_channels, modes1 = mode, modes2 = mode)

        self.litho_attn = ChunkedLithoAttn(compress_factor, source_d_model, value_d_model, z_channels * 2, mask_chunk_size, source_chunk_size)

        self.mask_down = Downsample(1, True)
        self.rfno_down = Downsample(z_channels, True)
        self.resist_up = Upsample(1, True)


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
                               z_channels = z_channels * 2,
                               double_z = False)

    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        rfno_branch: Tensor = self.rfno(F.avg_pool2d(mask, kernel_size = self.rfno_pool, stride = self.rfno_pool))
        rfno_embed: Tensor = self.rfno_down(rfno_branch)
        mask_down = self.mask_down(mask)
        mask_embed: Tensor = self.encoder(mask_down)
        mask_merge = torch.cat([rfno_embed, mask_embed], dim = 1)
        mask_attn: Tensor = self.litho_attn(source, mask_merge, dose, defocus)
        resist_embed = self.decoder(mask_attn)
        return self.resist_up(resist_embed)

if __name__ == '__main__':
    pass