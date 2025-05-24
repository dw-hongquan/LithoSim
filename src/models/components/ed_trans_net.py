from src.models.components.basic_net import ImageEncoderViT, Decoder, Downsample, Upsample
from src.models.components.litho_embed_net import ChunkedLithoAttn

from torch import nn, Tensor

class EDTransLitho(nn.Module):
    def __init__(self, scale_num: int = 1, z_channels: int = 64, compress_factor: int = 16, source_d_model: int = 8, value_d_model: int = 8, mask_chunk_size: int = 64, source_chunk_size: int = 256) -> None:
        super().__init__()

        self.down_block = nn.ModuleList()
        self.up_block = nn.ModuleList()

        self.litho_attn = ChunkedLithoAttn(compress_factor, source_d_model, value_d_model, z_channels, mask_chunk_size, source_chunk_size)

        for _ in range(scale_num):
            self.down_block.append(Downsample(1, True))
            self.up_block.append(Upsample(1, True))


        self.encoder = ImageEncoderViT(img_size = 2048,
                                       patch_size = 32,
                                       in_chans = 1,
                                       embed_dim = 512,
                                       depth = 16,
                                       num_heads = 16,
                                       mlp_ratio = 4.0,
                                       out_chans = z_channels,
                                       qkv_bias = True,
                                       use_rel_pos = False,
                                       rel_pos_zero_init = True,
                                       window_size = 0,
                                       global_attn_indexes = [])

        self.decoder = Decoder(ch = 64,
                               out_ch = 1,
                               ch_mult = [1, 1, 2, 2, 4, 4],
                               num_res_blocks = 1,
                               attn_resolutions = [16],
                               dropout = 0.2,
                               resamp_with_conv = True,
                               in_channels = 1,
                               resolution = 256,
                               z_channels = z_channels,
                               double_z = False)

    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        mask_down = mask
        for down in self.down_block:
            mask_down = down(mask_down)

        mask_embed: Tensor = self.encoder(mask_down)
        mask_attn: Tensor = self.litho_attn(source, mask_embed, dose, defocus)

        resist_embed = self.decoder(mask_attn)
        for up in self.up_block:
            resist_embed = up(resist_embed)

        return resist_embed
