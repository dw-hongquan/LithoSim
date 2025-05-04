'''
Author: Hongquan
Date: Apr. 22, 2025
Description: modules for torch.complex64.
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
        """
        频域优化的复数卷积层
        
        参数:
            in_channels: 输入通道数(复数视为1个通道)
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            dilation: 膨胀率
            groups: 分组数
            bias: 是否使用偏置
            padding_mode: 填充模式
            spectral_norm: 是否对权重应用谱归一化
            phase_correction: 是否应用相位校正
        """
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
        
        # 复数权重初始化 (实部和虚部)
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
            
        # 相位校正参数
        if phase_correction:
            self.phase_scale = nn.Parameter(torch.ones(1))
            self.phase_shift = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter('phase_scale', None)
            self.register_parameter('phase_shift', None)
            
        self.reset_parameters()
        
        # 谱归一化
        if spectral_norm:
            self._spectral_norm()

    def reset_parameters(self):
        # 复数权重初始化 (Glorot初始化)
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
        """应用谱归一化到权重矩阵"""
        with torch.no_grad():
            # 计算权重矩阵的奇异值
            weight_shape = self.weight_real.shape
            weight_matrix = torch.view_as_complex(
                torch.stack([self.weight_real, self.weight_imag], dim=-1))
            weight_matrix = weight_matrix.view(weight_shape[0], -1)
            
            # 幂迭代法估计最大奇异值
            u = torch.randn(weight_shape[0], 1, device=weight_matrix.device)
            v = torch.randn(1, weight_matrix.size(1), device=weight_matrix.device)
            
            for _ in range(3):  # 通常3次迭代足够
                v = F.normalize(torch.mm(u.t(), weight_matrix).t(), dim=0)
                u = F.normalize(torch.mm(weight_matrix, v), dim=0)
                
            sigma = torch.mm(torch.mm(u.t(), weight_matrix), v)
            
            # 归一化权重
            weight_matrix = weight_matrix / sigma
            
            # 重新分解为实部和虚部
            weight_complex = weight_matrix.view(*weight_shape)
            self.weight_real.data = weight_complex.real
            self.weight_imag.data = weight_complex.imag

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_complex():
            raise ValueError("Input must be a complex tensor")
            
        # 分离实部和虚部
        x_real = x.real
        x_imag = x.imag
        
        # 执行复数卷积 (实部卷积和虚部卷积)
        conv_real = F.conv2d(
            x_real, self.weight_real, None, self.stride, self.padding,
            self.dilation, self.groups)
        conv_imag = F.conv2d(
            x_imag, self.weight_imag, None, self.stride, self.padding,
            self.dilation, self.groups)
        
        # 交叉项
        conv_cross_real = F.conv2d(
            x_real, self.weight_imag, None, self.stride, self.padding,
            self.dilation, self.groups)
        conv_cross_imag = F.conv2d(
            x_imag, self.weight_real, None, self.stride, self.padding,
            self.dilation, self.groups)
        
        # 组合结果 (复数乘法)
        out_real = conv_real - conv_imag
        out_imag = conv_cross_real + conv_cross_imag
        
        # 应用相位校正
        if self.phase_correction:
            magnitude = torch.sqrt(out_real**2 + out_imag**2 + 1e-8)
            phase = torch.atan2(out_imag, out_real)
            
            # 可学习的相位调整
            phase = phase * self.phase_scale + self.phase_shift
            
            out_real = magnitude * torch.cos(phase)
            out_imag = magnitude * torch.sin(phase)
        
        # 添加偏置
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

def nonlinearity_1(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

def nonlinearity(x: Tensor) -> Tensor:
    '''
    nonlinearity for FFT complex tensor
    '''
    magnitude = torch.abs(x)
    phase = torch.angle(x)

    magnitude_out = magnitude * torch.sigmoid(magnitude)

    return magnitude_out * torch.exp(1j * phase)

class ComplexGroupNorm_2(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, 
                 phase_normalization=True, magnitude_scaling=True):
        """
        SpectralGroupNorm
        
        params:
            num_groups: 分组数量
            num_channels: 输入通道数(复数视为1个通道)
            eps: 数值稳定性常数
            affine: 是否使用可学习的缩放和平移参数
            phase_normalization: 是否对相位进行归一化
            magnitude_scaling: 是否对幅度进行缩放
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.phase_normalization = phase_normalization
        self.magnitude_scaling = magnitude_scaling

        if self.magnitude_scaling and affine:
            self.weight_mag = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias_mag = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight_mag', None)
            self.register_parameter('bias_mag', None)

        if self.phase_normalization and affine:
            self.weight_phase = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias_phase = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight_phase', None)
            self.register_parameter('bias_phase', None)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_complex():
            raise ValueError("Input must be a complex tensor.")
        
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or Inf values")

        B, C, H, W = x.shape
        G = self.num_groups
        

        magnitude = x.abs()
        phase = torch.atan2(x.imag, x.real)  # [-π, π]

        magnitude = magnitude.view(B, G, -1, H, W)
        

        mean_mag = magnitude.mean(dim=[2, 3, 4], keepdim=True)
        var_mag = magnitude.var(dim=[2, 3, 4], keepdim=True)
        

        magnitude_norm = (magnitude - mean_mag) / torch.sqrt(var_mag + self.eps)
        magnitude_norm = magnitude_norm.view(B, C, H, W)
        

        if self.magnitude_scaling and self.affine:
            magnitude_norm = magnitude_norm * self.weight_mag + self.bias_mag
        

        if self.phase_normalization:

            phase_cos = torch.cos(phase)
            phase_sin = torch.sin(phase)

            phase_cos = phase_cos.view(B, G, -1, H, W)
            phase_sin = phase_sin.view(B, G, -1, H, W)

            mean_cos = phase_cos.mean(dim=[2, 3, 4], keepdim=True)
            mean_sin = phase_sin.mean(dim=[2, 3, 4], keepdim=True)

            phase_cos_norm = phase_cos - mean_cos
            phase_sin_norm = phase_sin - mean_sin

            norm_length = torch.sqrt(phase_cos_norm**2 + phase_sin_norm**2 + self.eps)
            phase_cos_norm = phase_cos_norm / norm_length
            phase_sin_norm = phase_sin_norm / norm_length

            phase_norm = torch.atan2(phase_sin_norm, phase_cos_norm)
            phase_norm = phase_norm.view(B, C, H, W)

            if self.affine:
                phase_norm = phase_norm * self.weight_phase + self.bias_phase
        else:
            phase_norm = phase

        real_norm = magnitude_norm * torch.cos(phase_norm)
        imag_norm = magnitude_norm * torch.sin(phase_norm)
        
        return torch.complex(real_norm, imag_norm)
    
    def extra_repr(self):
        return (f"num_groups={self.num_groups}, num_channels={self.num_channels}, "
                f"eps={self.eps}, affine={self.affine}, "
                f"phase_normalization={self.phase_normalization}, "
                f"magnitude_scaling={self.magnitude_scaling}")

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

class ComplexGroupNorm_1(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=2*num_channels, eps=eps, affine=affine)

    def forward(self, x: Tensor) -> Tensor:
        x_real = torch.view_as_real(x)
        
        original_shape = x_real.size()
        x_combined = x_real.permute(0, 1, 4, 2, 3).contiguous()  # 将最后的2维移到通道后
        x_combined = x_combined.view(original_shape[0], -1, *original_shape[2:-1])  # 合并C和2到通道维度

        x_normalized = self.gn(x_combined)
        x_normalized = x_normalized.view(original_shape[0], self.num_channels, 2, *original_shape[2:-1])
        x_normalized = x_normalized.permute(0, 1, 3, 4, 2).contiguous()  # 恢复维度顺序
        x_complex = torch.view_as_complex(x_normalized)
        
        return x_complex

def ComplexNormalize(in_channels: int, num_groups: int = 32):
    assert in_channels % num_groups == 0, f'in_channels {in_channels} must be devided by num_groups: {num_groups}'
    return ComplexGroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class ComplexUpsample(nn.Module):
    def __init__(self, scale_factor: int = 2, mode: str = 'bilinear', 
                 phase_mode: str = 'bilinear', align_corners: bool = False) -> None:
        """
        PolarSampling
        
        params:
            scale_factor: 上采样比例因子
            mode: 幅度上采样模式 ('nearest', 'bilinear', 'bicubic')
            phase_mode: 相位上采样模式 ('nearest', 'bilinear', 'bicubic')
            align_corners: 是否对齐角落像素
        """
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

class ComplexUpsample_1(nn.Module):
    def __init__(self, scale_factor: int = 2, mode: str = 'nearest', align_corners: bool = None):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners if mode != 'nearest' else None

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_complex():
            raise ValueError("Input must be a complex tensor.")
        

        x_real = torch.view_as_real(x)

        N, C, H, W = x.size()
        x_real = x_real.permute(0, 1, 4, 2, 3).reshape(N, C * 2, H, W)

        upsampled = F.interpolate(
            x_real, 
            scale_factor=self.scale_factor, 
            mode=self.mode, 
            align_corners=self.align_corners
        )

        new_C = upsampled.size(1) // 2
        upsampled = upsampled.view(N, new_C, 2, upsampled.size(2), upsampled.size(3)).contiguous()
        upsampled = upsampled.permute(0, 1, 3, 4, 2).contiguous()

        return torch.view_as_complex(upsampled)

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
        """
        PolarAvgPool2d
        
        params:
            kernel_size: 池化核大小
            stride: 步长
            padding: 填充
            ceil_mode: 是否使用ceil模式计算输出形状
            count_include_pad: 是否包含padding在平均计算中
            divisor_override: 可选的除数覆盖
            phase_pool_mode: 相位池化模式 ('circular' 或 'linear')
                            'circular' - 在单位圆上平均相位
                            'linear' - 直接线性平均相位(不推荐)
        """
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

class ComplexAvgPool2d_1(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0,
                 ceil_mode=False, count_include_pad=True, divisor_override=None):
        super().__init__()
        self.avgpool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override
        )
    
    def forward(self, x):
        assert x.is_complex(), "Input must be a complex tensor"
        real = self.avgpool(x.real)
        imag = self.avgpool(x.imag)
        return torch.complex(real, imag)

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
        """
        SpectralDropout
        
        params:
            p: dropout概率
            mode: dropout模式 ('magnitude'或'complex')
                  'magnitude' - 只对幅度进行dropout
                  'complex' - 对整个复数进行dropout
            phase_preserve: 是否保持相位关系(仅当mode='magnitude'时有效)
        """
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

class ComplexDropout_1(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), but got {}".format(p))
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        keep_prob = 1 - self.p
        scale = 1 / keep_prob
        mask = torch.rand_like(x.real) < keep_prob
        mask = mask.type_as(x.real)
        mask = mask * scale

        return x * mask

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
        """
        SpectralSoftmax
        
        参数:
            dim: 计算softmax的维度
            temperature: 温度参数控制softmax的锐度
            phase_preserve: 是否保持原始相位信息
            eps: 数值稳定性常数
        """
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

class ComplexSoftmax_1(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
        self.softmax = nn.Softmax(dim=dim)
    
    def forward(self, input: Tensor) -> Tensor:
        real_part = input.real
        softmax_real = self.softmax(real_part)
        phase = torch.angle(input)
        output = torch.polar(softmax_real, phase)
        return output

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
