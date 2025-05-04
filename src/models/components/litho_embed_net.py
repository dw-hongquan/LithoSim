'''
Author: Hongquan
Date: Apr. 18, 2025
'''

import sys
sys.path.append('.')

import torch
from torch import nn, Tensor, einsum
from src.models.components.pe_net import continuous_positional_encoding_1d, continuous_positional_encoding_2d
from src.models.components.attention_net import ValueCrossAttn, PosCrossAttn, ChunkedValueCrossAttn, ChunkedPosCrossAttn, ComplexChunkedValueCrossAttn, ComplexChunkedPosCrossAttn
from src.models.components.complex_net import ComplexGroupNorm

def source_encoding(source: Tensor, d_model: int = 64) -> Tensor:
    """
        :source: [batch_size, N, 3]
        :return: [batch_size, N, d_model]
    """
    coords = source[..., :2].contiguous()
    values = source[..., 2:].contiguous()
    pos_emb = continuous_positional_encoding_2d(d_model, coords)
    return torch.cat([pos_emb, values], dim = -1).contiguous()

def value_encoding(values: Tensor, d_model: int = 16) -> Tensor:
    """
        :values: [batch_size, 1]
        :return: [batch_size, d_model]
    """
    return continuous_positional_encoding_1d(d_model, values).contiguous()


class SourceCompressor(nn.Module):
    def __init__(self, K: int, context_dim: int = 65):
        super().__init__()
        self.K = K
        self.scale = context_dim ** -0.5
        # [K, context_dim]
        self.queries = nn.Parameter(torch.randn(K, context_dim) * self.scale)

        self.key_layer = nn.Linear(context_dim, context_dim)
        self.value_layer = nn.Linear(context_dim, context_dim)
        nn.init.normal_(self.key_layer.weight, mean=0, std=self.scale)

    def forward(self, x: Tensor) -> Tensor:
        """
            :x: [B, N, context_dim], context_dim = D
            :return: [B, K, context_dim]
        """
        B = x.shape[0]

        keys = self.key_layer(x) # [B, N, D]
        values = self.value_layer(x) # [B, N, D]
        # [B, K, D]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        attn_scores = torch.matmul(queries, keys.transpose(1,2).contiguous()) * self.scale # [B, K, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_weights, values) # [B, K, D]
        return output.contiguous()

class ChunkedSourceCompressor(nn.Module):
    def __init__(self, K: int, context_dim: int = 65, chunk_size: int = 64, 
                 attn_dropout: float = 0.1) -> None:
        super().__init__()
        self.scale = context_dim ** -0.5
        self.K = K
        self.chunk_size = chunk_size

        self.query_gen = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, K * context_dim)
        )

        self.key_proj = nn.Linear(context_dim, context_dim)
        self.value_proj = nn.Linear(context_dim, context_dim)

        self.cross_block_query = nn.Parameter(torch.randn(1, context_dim) * self.scale)

        self.pos_enc = nn.Parameter(torch.randn(1, chunk_size, context_dim) * self.scale)

        nn.init.normal_(self.key_proj.weight, mean=0, std=self.scale)
        nn.init.normal_(self.value_proj.weight, mean=0, std=self.scale)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        for layer in [self.query_gen[0], self.query_gen[2]]:
            nn.init.normal_(layer.weight, mean=0, std=self.scale)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.cross_block_query, mean=0, std=self.scale)
        nn.init.normal_(self.pos_enc, mean=0, std=self.scale)

        self.activation = nn.GELU()

        self.norm = nn.LayerNorm(context_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

        self.res_proj = nn.Linear(context_dim, context_dim)
        nn.init.zeros_(self.res_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        chunk_size = self.chunk_size

        original_mask = None
        if N % chunk_size != 0:
            pad = chunk_size - (N % chunk_size)
            x = nn.functional.pad(x, (0, 0, 0, pad))
            original_mask = torch.zeros(B, N + pad, dtype=torch.bool, device=x.device)
            original_mask[:, :N] = True
        else:
            pad = 0
            original_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        
        N_padded = x.size(1)
        num_blocks = N_padded // chunk_size

        sum_x = (x * original_mask.unsqueeze(-1)).sum(dim=1)
        count = original_mask.sum(dim=1, keepdim=True).clamp(min=1e-9)
        global_avg = sum_x / count

        queries = self.query_gen(global_avg).view(B, self.K, D).contiguous()
        queries = queries * self.scale

        keys = self.activation(self.key_proj(x))
        values = self.activation(self.value_proj(x))

        keys = keys.view(B, num_blocks, chunk_size, D).contiguous() + self.pos_enc
        keys = keys.view(B, -1, D).contiguous()
        values = values.view(B, num_blocks, chunk_size, D).contiguous() + self.pos_enc
        values = values.view(B, -1, D).contiguous()

        keys = keys.view(B, num_blocks, chunk_size, D).contiguous()
        values = values.view(B, num_blocks, chunk_size, D).contiguous()
        mask = original_mask.view(B, num_blocks, chunk_size).contiguous()

        attn_scores = torch.einsum('bkd,bncd->bnkc', queries, keys) * self.scale
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(2), float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        block_outputs = torch.einsum('bnkc,bncd->bnkd', attn_weights, values)

        block_outputs = block_outputs.transpose(1, 2).contiguous()  # [B, K, num_blocks, D]
        cross_query = self.cross_block_query.unsqueeze(0).unsqueeze(1).expand(B, self.K, -1, -1)
        cross_attn_scores = torch.matmul(cross_query, block_outputs.transpose(-2, -1).contiguous()) * self.scale
        cross_attn_weights = torch.softmax(cross_attn_scores, dim=-1)
        compressed = torch.matmul(cross_attn_weights, block_outputs).squeeze(2)

        residual = self.res_proj(global_avg).unsqueeze(1).expand(-1, self.K, -1)
        compressed = compressed + residual
        
        return self.norm(compressed)

class LithoAttn(nn.Module):
    def __init__(self, compress_factor: int = 32, source_d_model: int = 32, value_d_model: int = 16, query_dim: int = 64, num_norm_group: int = 8):
        super().__init__()
        assert query_dim % num_norm_group == 0, f"query_dim must be divisible by {num_norm_group}, got query_dim:{query_dim}"
        self.source_d_model = source_d_model
        self.value_d_model = value_d_model
        self.source_compressor = nn.Sequential(
            SourceCompressor(2048, self.source_d_model + 1),
            SourceCompressor(512, self.source_d_model + 1),
            SourceCompressor(128, self.source_d_model + 1),
            SourceCompressor(compress_factor, self.source_d_model + 1)
        )
        self.fused_norm = nn.GroupNorm(num_norm_group, query_dim)
        self.pos_attn = PosCrossAttn(query_dim = query_dim, context_dim = self.source_d_model + 1)
        self.dose_attn = ValueCrossAttn(query_dim = query_dim, context_dim = self.value_d_model)
        self.defocus_attn = ValueCrossAttn(query_dim = query_dim, context_dim = self.value_d_model)
    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        source_embed = source_encoding(source, self.source_d_model)
        dose_embed = value_encoding(dose, self.value_d_model)
        defocus_embed = value_encoding(defocus, self.value_d_model)
        compressed_source = self.source_compressor(source_embed)
        source_fused = self.pos_attn(mask, compressed_source).contiguous()
        dose_fused = self.dose_attn(mask, dose_embed).contiguous()
        defocus_fused = self.defocus_attn(mask, defocus_embed).contiguous()
        return self.fused_norm((source_fused + dose_fused + defocus_fused).contiguous())

class ChunkedLithoAttn(nn.Module):
    def __init__(self, compress_factor: int = 32, source_d_model: int = 32, value_d_model: int = 16, query_dim: int = 64, mask_chunk_size: int = 64, source_chunk_size: int = 256, num_norm_group: int = 8):
        super().__init__()
        assert query_dim % num_norm_group == 0, f"query_dim must be divisible by {num_norm_group}, got query_dim:{query_dim}"
        self.source_d_model = source_d_model
        self.value_d_model = value_d_model
        self.source_compressor = ChunkedSourceCompressor(compress_factor, self.source_d_model + 1, source_chunk_size)

        self.fused_norm = nn.GroupNorm(num_norm_group, query_dim)
        self.pos_attn = ChunkedPosCrossAttn(query_dim = query_dim, context_dim = self.source_d_model + 1, chunk_size = mask_chunk_size)
        self.dose_attn = ChunkedValueCrossAttn(query_dim = query_dim, context_dim = self.value_d_model, chunk_size = mask_chunk_size)
        self.defocus_attn = ChunkedValueCrossAttn(query_dim = query_dim, context_dim = self.value_d_model, chunk_size = mask_chunk_size)
    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        source_embed = source_encoding(source, self.source_d_model)
        dose_embed = value_encoding(dose, self.value_d_model)
        defocus_embed = value_encoding(defocus, self.value_d_model)
        compressed_source = self.source_compressor(source_embed)

        source_fused = self.pos_attn(mask, compressed_source).contiguous()
        
        dose_fused = self.dose_attn(mask, dose_embed).contiguous()

        defocus_fused = self.defocus_attn(mask, defocus_embed).contiguous()

        return self.fused_norm((source_fused + dose_fused + defocus_fused).contiguous())

class ComplexChunkedLithoAttn(nn.Module):
    def __init__(self, compress_factor: int = 32, source_d_model: int = 32, value_d_model: int = 16, query_dim: int = 64, mask_chunk_size: int = 64, source_chunk_size: int = 256, num_norm_group: int = 8):
        super().__init__()
        assert query_dim % num_norm_group == 0, f"query_dim must be divisible by {num_norm_group}, got query_dim:{query_dim}"
        self.source_d_model = source_d_model
        self.value_d_model = value_d_model
        self.source_compressor = ChunkedSourceCompressor(compress_factor, self.source_d_model + 1, source_chunk_size)

        self.fused_norm = ComplexGroupNorm(num_norm_group, query_dim)
        self.pos_attn = ComplexChunkedPosCrossAttn(query_dim = query_dim, context_dim = self.source_d_model + 1, chunk_size = mask_chunk_size)
        self.dose_attn = ComplexChunkedValueCrossAttn(query_dim = query_dim, context_dim = self.value_d_model, chunk_size = mask_chunk_size)
        self.defocus_attn = ComplexChunkedValueCrossAttn(query_dim = query_dim, context_dim = self.value_d_model, chunk_size = mask_chunk_size)
    def forward(self, source: Tensor, mask: Tensor, dose: Tensor, defocus: Tensor) -> Tensor:
        source_embed = source_encoding(source, self.source_d_model)
        dose_embed = value_encoding(dose, self.value_d_model)
        defocus_embed = value_encoding(defocus, self.value_d_model)
        compressed_source = self.source_compressor(source_embed)

        source_fused = self.pos_attn(mask, compressed_source).contiguous()
        
        dose_fused = self.dose_attn(mask, dose_embed).contiguous()

        defocus_fused = self.defocus_attn(mask, defocus_embed).contiguous()

        return self.fused_norm((source_fused + dose_fused + defocus_fused).contiguous())


if __name__ == "__main__":
    # # 输入示例 [B=2, N=7180, D=65]
    # input_tensor = torch.randn(2, 7180, 65)
    
    # # 初始化压缩器（将N从7180压缩到2048）
    # compressor = SourceCompressor(K=2048)
    
    # # 前向传播
    # output = compressor(input_tensor)
    
    # print("输入形状:", input_tensor.shape)  # torch.Size([2, 7180, 65])
    # print("输出形状:", output.shape)       # torch.Size([2, 2048, 65])

    compressor = ChunkedSourceCompressor(K=32, context_dim=64, chunk_size=128)
    x = torch.randn(4, 10000, 64)  # 假设输入包含10000个元素
    output = compressor(x)          # 输出形状: [4, 32, 64]
    print(output.shape)
