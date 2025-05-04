'''
Author: Hongquan
Date: Apr. 24, 2025
Description: Encoder-Decoder CNN for litho.
'''

from torch import nn, Tensor
from src.models.components.litho_embed_net import ChunkedLithoAttn
from src.models.components.basic_net import Downsample, Upsample, Encoder, Decoder

class EDCNNLitho(nn.Module):
    def __init__(self, z_channels: int = 64, compress_factor: int = 16, source_d_model: int = 8, value_d_model: int = 8, mask_chunk_size: int = 64, source_chunk_size: int = 256) -> None:
        super().__init__()

        self.litho_attn = ChunkedLithoAttn(compress_factor, source_d_model, value_d_model, z_channels, mask_chunk_size, source_chunk_size)

        self.mask_down = Downsample(1, True)
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
                               z_channels = z_channels,
                               double_z = False)

    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        mask_down = self.mask_down(mask)
        mask_embed: Tensor = self.encoder(mask_down)
        mask_attn: Tensor = self.litho_attn(source, mask_embed, dose, defocus)
        resist_embed = self.decoder(mask_attn)
        return self.resist_up(resist_embed)

if __name__ == '__main__':
    pass