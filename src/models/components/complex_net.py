'''
Author: Hongquan
Date: Apr. 22, 2025
Description: Complex-valued edcoder and decoder for LithoSim
'''
import sys
sys.path.append('.')

from torch import nn, Tensor, complex64
import torch
import torch.nn.functional as F
import math


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 spectral_norm=False, phase_correction=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.spectral_norm = spectral_norm
        self.phase_correction = phase_correction
        
        self.weight_real = nn.Parameter(torch.empty(
            (out_channels, in_channels // groups, *self.kernel_size),
            dtype=torch.float32))
        self.weight_imag = nn.Parameter(torch.empty(
            (out_channels, in_channels // groups, *self.kernel_size),
            dtype=torch.float32))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
            
        if phase_correction:
            self.phase_scale = nn.Parameter(torch.ones(1))
            self.phase_shift = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('phase_scale', None)
            self.register_parameter('phase_shift', None)
            
        self.reset_parameters()
        
        if spectral_norm:
            self._spectral_norm()

    def reset_parameters(self):
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        fan_out = self.out_channels * self.kernel_size[0] * self.kernel_size[1]
        
        std_real = math.sqrt(2.0 / (fan_in + fan_out))
        std_imag = math.sqrt(2.0 / (fan_in + fan_out))
        
        nn.init.normal_(self.weight_real, 0.0, std_real)
        nn.init.normal_(self.weight_imag, 0.0, std_imag)
        
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            
    def _spectral_norm(self):
        with torch.no_grad():
            weight_shape = self.weight_real.shape
            weight_matrix = torch.view_as_complex(
                torch.stack([self.weight_real, self.weight_imag], dim=-1))
            weight_matrix = weight_matrix.view(weight_shape[0], -1)
            
            u = torch.randn(weight_shape[0], 1, device=weight_matrix.device)
            v = torch.randn(1, weight_matrix.size(1), device=weight_matrix.device)
            
            for _ in range(3):
                v = F.normalize(torch.mm(u.t(), weight_matrix).t(), dim=0)
                u = F.normalize(torch.mm(weight_matrix, v), dim=0)
                
            sigma = torch.mm(torch.mm(u.t(), weight_matrix), v)

            weight_matrix = weight_matrix / sigma

            weight_complex = weight_matrix.view(*weight_shape)
            self.weight_real.data = weight_complex.real
            self.weight_imag.data = weight_complex.imag

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_complex():
            raise ValueError("Input must be a complex tensor")

        x_real = x.real
        x_imag = x.imag

        conv_real = F.conv2d(
            x_real, self.weight_real, None, self.stride, self.padding,
            self.dilation, self.groups)
        conv_imag = F.conv2d(
            x_imag, self.weight_imag, None, self.stride, self.padding,
            self.dilation, self.groups)

        conv_cross_real = F.conv2d(
            x_real, self.weight_imag, None, self.stride, self.padding,
            self.dilation, self.groups)
        conv_cross_imag = F.conv2d(
            x_imag, self.weight_real, None, self.stride, self.padding,
            self.dilation, self.groups)

        out_real = conv_real - conv_imag
        out_imag = conv_cross_real + conv_cross_imag

        if self.phase_correction:
            magnitude = torch.sqrt(out_real**2 + out_imag**2 + 1e-8)
            phase = torch.atan2(out_imag, out_real)
            

            phase = phase * self.phase_scale + self.phase_shift
            
            out_real = magnitude * torch.cos(phase)
            out_imag = magnitude * torch.sin(phase)
        

        if self.bias is not None:
            out_real = out_real + self.bias.view(1, -1, 1, 1)
            out_imag = out_imag + self.bias.view(1, -1, 1, 1)
            
        return torch.complex(out_real, out_imag)
    
    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        if self.spectral_norm:
            s += ', spectral_norm=True'
        if self.phase_correction:
            s += ', phase_correction=True'
        return s.format(**self.__dict__)

def nonlinearity(x: Tensor) -> Tensor:
    '''
    nonlinearity for FFT complex tensor
    '''
    magnitude = torch.abs(x)
    phase = torch.angle(x)

    magnitude_out = magnitude * torch.sigmoid(magnitude)

    return magnitude_out * torch.exp(1j * phase)

class ComplexGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        self.gn = nn.GroupNorm(
            num_groups=num_groups, 
            num_channels=2 * num_channels, 
            eps=eps, 
            affine=affine
        )

    def forward(self, x: Tensor) -> Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"ComplexGroupNorm contain NaN/Inf")

        x_real = torch.view_as_real(x)  # [B, C, H, W, 2]
        original_shape = x_real.shape

        x_combined = x_real.permute(0, 1, 4, 2, 3).contiguous()  # [B, C, 2, H, W]
        x_combined = x_combined.view(original_shape[0], -1, *original_shape[2:-1])  # [B, 2*C, H, W]

        x_normalized = self.gn(x_combined)

        x_normalized = x_normalized.view(original_shape[0], self.num_channels, 2, *original_shape[2:-1]).contiguous()  # [B, C, 2, H, W]
        x_normalized = x_normalized.permute(0, 1, 3, 4, 2).contiguous()
        x_complex = torch.view_as_complex(x_normalized)

        return x_complex


def ComplexNormalize(in_channels: int, num_groups: int = 32):
    assert in_channels % num_groups == 0, f'in_channels {in_channels} must be devided by num_groups: {num_groups}'
    return ComplexGroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ComplexUpsample(nn.Module):
    def __init__(self, scale_factor: int = 2, mode: str = 'bilinear', 
                 phase_mode: str = 'bilinear', align_corners: bool = False) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.phase_mode = phase_mode
        self.align_corners = align_corners
        if phase_mode not in ['nearest', 'bilinear', 'bicubic']:
            raise ValueError("phase_mode must be 'nearest', 'bilinear' or 'bicubic'")

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_complex():
            raise ValueError("Input must be a complex tensor.")
        
        magnitude = x.abs()
        phase = torch.atan2(x.imag, x.real)

        up_magnitude = F.interpolate(
            magnitude,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners if self.mode != 'nearest' else None
        )
        
        phase_cos = torch.cos(phase)
        phase_sin = torch.sin(phase)
        
        up_cos = F.interpolate(
            phase_cos,
            scale_factor=self.scale_factor,
            mode=self.phase_mode,
            align_corners=self.align_corners if self.phase_mode != 'nearest' else None
        )
        
        up_sin = F.interpolate(
            phase_sin,
            scale_factor=self.scale_factor,
            mode=self.phase_mode,
            align_corners=self.align_corners if self.phase_mode != 'nearest' else None
        )

        up_phase = torch.atan2(up_sin, up_cos)

        up_real = up_magnitude * torch.cos(up_phase)
        up_imag = up_magnitude * torch.sin(up_phase)
        
        return torch.complex(up_real, up_imag)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(scale_factor={self.scale_factor}, "
                f"magnitude_mode={self.mode}, phase_mode={self.phase_mode}, "
                f"align_corners={self.align_corners})")

class ComplexConvUpsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  dtype=complex64
                                  )
        self.upsample = ComplexUpsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        x = self.upsample(x)
        if self.with_conv:
            x = self.conv(x)
        return x

class ComplexAvgPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, 
                 divisor_override=None, phase_pool_mode='circular'):

        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.phase_pool_mode = phase_pool_mode

        self.magnitude_pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_complex():
            raise ValueError("Input must be a complex tensor.")

        magnitude = x.abs()
        phase = torch.atan2(x.imag, x.real)

        pooled_magnitude = self.magnitude_pool(magnitude)

        if self.phase_pool_mode == 'circular':

            phase_cos = torch.cos(phase)
            phase_sin = torch.sin(phase)

            pooled_cos = self.magnitude_pool(phase_cos)
            pooled_sin = self.magnitude_pool(phase_sin)

            pooled_phase = torch.atan2(pooled_sin, pooled_cos)
        elif self.phase_pool_mode == 'linear':

            phase_unwrapped = self.unwrap_phase(phase)
            pooled_phase_unwrapped = self.magnitude_pool(phase_unwrapped)
            pooled_phase = self.wrap_phase(pooled_phase_unwrapped)
        else:
            raise ValueError(f"Unknown phase_pool_mode: {self.phase_pool_mode}")
        
        pooled_real = pooled_magnitude * torch.cos(pooled_phase)
        pooled_imag = pooled_magnitude * torch.sin(pooled_phase)
        
        return torch.complex(pooled_real, pooled_imag)
    
    @staticmethod
    def unwrap_phase(phase):
        diff = torch.zeros_like(phase)
        diff[1:] = phase[1:] - phase[:-1]
        phase_unwrapped = phase.clone()
        phase_unwrapped[1:] += -2 * math.pi * torch.floor((diff[1:] + math.pi) / (2 * math.pi))
        return phase_unwrapped
    
    @staticmethod
    def wrap_phase(phase):
        return (phase + math.pi) % (2 * math.pi) - math.pi
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, "
                f"ceil_mode={self.ceil_mode}, count_include_pad={self.count_include_pad}, "
                f"divisor_override={self.divisor_override}, phase_pool_mode={self.phase_pool_mode})")

class ComplexConvDownsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        dtype=complex64
                                        )
        else:
            self.avg_pool2d = ComplexAvgPool2d(kernel_size=2, stride=2)
    def forward(self, x: Tensor) -> Tensor:
        if self.with_conv:
            x = self.conv(x)
        else:
            x = self.avg_pool2d(x)
        return x

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5, mode='magnitude', phase_preserve=True):

        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.mode = mode
        self.phase_preserve = phase_preserve

        if mode not in ['magnitude', 'complex']:
            raise ValueError("mode must be either 'magnitude' or 'complex'")
        if mode == 'complex' and phase_preserve:
            raise ValueError("phase_preserve cannot be True when mode='complex'")

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_complex():
            raise ValueError("Input must be a complex tensor")
            
        if not self.training or self.p == 0:
            return x
            
        keep_prob = 1 - self.p
        scale = 1 / keep_prob
        
        if self.mode == 'magnitude':
            magnitude = x.abs()
            phase = torch.atan2(x.imag, x.real)
            mask = torch.rand_like(magnitude) < keep_prob
            mask = mask.type_as(magnitude) * scale

            new_magnitude = magnitude * mask

            if self.phase_preserve:
                return torch.polar(new_magnitude, phase)
            else:
                phase_noise = torch.rand_like(phase) * 0.01
                return torch.polar(new_magnitude, phase + phase_noise)
                
        else:  # mode == 'complex'
            mask_real = torch.rand_like(x.real) < keep_prob
            mask_imag = torch.rand_like(x.imag) < keep_prob
            
            if self.training:
                mask = mask_real & mask_imag
            else:
                mask = mask_real
            
            mask = mask.type_as(x.real) * scale
            return x * mask
            
    def extra_repr(self):
        return f"p={self.p}, mode='{self.mode}', phase_preserve={self.phase_preserve}"


class ComplexResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = ComplexNormalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               dtype=complex64
                               )
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels,
                                       out_channels,
                                    #    dtype=complex64
                                       )
        self.norm2 = ComplexNormalize(out_channels)
        self.dropout = ComplexDropout(dropout)
        self.conv2 = nn.Conv2d(out_channels,
                               out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               dtype=complex64
                               )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels,
                                               out_channels,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1,
                                               dtype=complex64
                                               )
            else:
                self.nin_shortcut = nn.Conv2d(in_channels,
                                              out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              dtype=complex64
                                              )

    def forward(self, x: Tensor, temb=None) -> Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class ComplexSoftmax(nn.Module):
    def __init__(self, dim: int = -1, temperature: float = 1.0, 
                 phase_preserve: bool = True, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.phase_preserve = phase_preserve
        self.eps = eps

    def forward(self, input: Tensor) -> Tensor:
        if not input.is_complex():
            raise ValueError("Input must be a complex tensor")

        magnitude = input.abs()
        phase = torch.atan2(input.imag, input.real)
        
        magnitude_scaled = magnitude / self.temperature
        max_magnitude = magnitude_scaled.max(dim=self.dim, keepdim=True).values
        magnitude_exp = torch.exp(magnitude_scaled - max_magnitude)
        magnitude_softmax = magnitude_exp / (magnitude_exp.sum(dim=self.dim, keepdim=True) + self.eps)

        if self.phase_preserve:
            new_phase = phase
        else:
            phase_cos = torch.cos(phase)
            phase_sin = torch.sin(phase)
            R = torch.sqrt(phase_sin.pow(2).sum(dim=self.dim, keepdim=True) + 
                          phase_cos.pow(2).sum(dim=self.dim, keepdim=True)) / phase.size(self.dim)
            
            phase_mean = torch.atan2(phase_sin.sum(dim=self.dim, keepdim=True),
                                   phase_cos.sum(dim=self.dim, keepdim=True))
            new_phase = phase_mean * (1 - R) + phase * R
        
        output_real = magnitude_softmax * torch.cos(new_phase)
        output_imag = magnitude_softmax * torch.sin(new_phase)
        
        return torch.complex(output_real, output_imag)
    
    def extra_repr(self):
        return (f"dim={self.dim}, temperature={self.temperature}, "
                f"phase_preserve={self.phase_preserve}, eps={self.eps}")

class ComplexAttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = ComplexNormalize(in_channels)
        self.softmax = ComplexSoftmax(dim=2)
        self.q = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           dtype=complex64
                           )
        self.k = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           dtype=complex64
                           )
        self.v = nn.Conv2d(in_channels,
                           in_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0,
                           dtype=complex64
                           )
        self.proj_out = nn.Conv2d(in_channels,
                                  in_channels,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  dtype=complex64
                                  )

    def forward(self, x: Tensor) -> Tensor:
        h_ = x.contiguous()
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w).contiguous()
        q = q.permute(0,2,1).contiguous()
        k = k.reshape(b,c,h*w).contiguous()
        w_ = torch.bmm(q,k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = self.softmax(w_)

        v = v.reshape(b,c,h*w).contiguous()
        w_ = w_.permute(0,2,1).contiguous()
        h_ = torch.bmm(v,w_)
        h_ = h_.reshape(b,c,h,w).contiguous()

        h_ = self.proj_out(h_)

        return x+h_

class ComplexEncoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, double_z=True, **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nn.Conv2d(in_channels,
                                 self.ch,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 dtype=complex64
                                 )

        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks): # i_block
                block.append(ComplexResnetBlock(in_channels=block_in,
                                                out_channels=block_out,
                                                temb_channels=self.temb_ch,
                                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(ComplexAttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = ComplexConvDownsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ComplexResnetBlock(in_channels=block_in,
                                              out_channels=block_in,
                                              temb_channels=self.temb_ch,
                                              dropout=dropout)
        self.mid.attn_1 = ComplexAttnBlock(block_in)
        self.mid.block_2 = ComplexResnetBlock(in_channels=block_in,
                                              out_channels=block_in,
                                              temb_channels=self.temb_ch,
                                              dropout=dropout)

        # end
        self.norm_out = ComplexNormalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  2*z_channels if double_z else z_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  dtype=complex64
                                  )


    def forward(self, x: Tensor) -> Tensor:
        #assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

class ComplexDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True, in_channels,
                 resolution, z_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        # in_ch_mult = (1,)+tuple(ch_mult)
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)
        # print("Working with z of shape {} = {} dimensions.".format(
        #     self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels,
                                 block_in,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 dtype=complex64
                                 )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ComplexResnetBlock(in_channels=block_in,
                                              out_channels=block_in,
                                              temb_channels=self.temb_ch,
                                              dropout=dropout)
        self.mid.attn_1 = ComplexAttnBlock(block_in)
        self.mid.block_2 = ComplexResnetBlock(in_channels=block_in,
                                              out_channels=block_in,
                                              temb_channels=self.temb_ch,
                                              dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for _ in range(self.num_res_blocks+1): # i_block
                block.append(ComplexResnetBlock(in_channels=block_in,
                                                out_channels=block_out,
                                                temb_channels=self.temb_ch,
                                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(ComplexAttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = ComplexConvUpsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = ComplexNormalize(block_in)
        self.conv_out = nn.Conv2d(block_in,
                                  out_ch,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  dtype=complex64
                                  )

    def forward(self, z: Tensor) -> Tensor:
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

if __name__ == '__main__':
    pass
