#%%

from typing import Tuple
import torch
from torch import FloatTensor, nn
import math
from dataclasses import dataclass


def calculate_out_hw(hw: int, k: int, s: int, p=0) -> int:
    """Calculates output hw given input hw, kernel size, stride, padding"""
    return math.floor(((hw + 2*p - k)/s)+1)

def calculate_out_hw_transpose(hw: int, k: int, s: int, p=0) -> int:
    """Transpose of calculate_out_hw function"""
    return ((hw - 1) * s) - (2 * p) + (k - 1) + 1

def get_best_ksp(in_hw: int, out_hw: int, max_kern=4, max_stride=3, max_pad=3, validfn=calculate_out_hw) -> Tuple[bool, Tuple[int, int, int]]:
    """
    Retrieves best (kernel, stride, padding) combination out of all valid combinations,
    in order to scale from `in_hw` to `out_hw`. Validity is determined by the given
    `validfn` - this will ensure the chosen (`k`, `s`, `p`) combinations all scale from
    `in_hw` to `out_hw`. Out of all valid combinations, we choose the best (`k`, `s`, `p`)
    such that `k` is largest, followed by `p`, followed by `s`.
    """
    kernels = range(1, max_kern+1)
    strides = range(1, max_stride+1)
    pads    = range(0, max_pad+1)
    params  = ((k,s,p) for k in kernels for s in strides for p in pads if validfn(in_hw, k, s, p) == out_hw)
    (k, s, p) = max(params, key=lambda x: (x[0], x[2], x[1]))
    try:
        return (k, s, p)
    except:
        raise Exception('Could not find valid parameters to produce output hw')

def convKxK(in_channels: int, out_channels: int, in_hw: int, out_hw: int, max_kern=4, max_stride=3, max_pad=2, bias=False) -> nn.Conv2d:
    """Generic function to create single flexible Conv2d OR Conv2dTraspose layer that will scale from in_hw to out_hw.
    If (in_hw > out_hw), will return a nn.Conv2d layer, else will return nn.ConvTranspose2d layer.
    """
    validfn = calculate_out_hw if (in_hw > out_hw) else calculate_out_hw_transpose
    (k, s, p) = get_best_ksp(in_hw, out_hw, max_kern, max_stride, max_pad, validfn=validfn)
    if (in_hw > out_hw):
        return nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=bias)
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=bias)

def conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    """3x3 Conv, leaves spatial dimensions unchanged"""
    return nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1, bias=False)

def conv3x3_relu_batchnorm(in_ch: int, out_ch: int, reluleak=0.2) -> nn.Sequential:
    """3x3 Conv followed by LeakyRelu then BatchNorm layer. Leaves spatial dimensions unchanged"""
    return nn.Sequential(
        conv3x3(in_ch, out_ch),
        nn.LeakyReLU(reluleak),
        nn.BatchNorm2d(out_ch))

def convdownblock(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 Conv followed by LeakyRelu & MaxPool. Halves spatial dimensions"""
    return nn.Sequential(
        conv3x3_relu_batchnorm(in_ch=in_ch, out_ch=out_ch),
        nn.MaxPool2d(kernel_size=2, stride=2))

def convupblock(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        conv3x3_relu_batchnorm(in_ch=in_ch, out_ch=out_ch))


class GCU(nn.Module):
    """
    Gated Convolutional Unit.
    Applies the function (X * W1_T + b1) * sigmoid(X * W2_T + b2).
    """
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)
        self.conv_gate = conv3x3(in_ch, out_ch)
        self.bnorm = nn.BatchNorm2d(num_features=out_ch)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """
        Args:
            x: ImageTensor, shape of (batch, self.in_ch, H, W)

        Output:
            ImageTensor with same spatial dimensions, shape of (batch, self.out_ch, H, W)
        """
        x_conv = self.conv(x)
        x_gate = self.conv_gate(x)
        return self.bnorm(x_conv * torch.sigmoid(x_gate))


class ResBlock(nn.Module):
    """
    Residual block - performs 2 sets of (conv3x3+relu+batchnorm)
    with a GCU sandwiched in between them, then adds the result to the original x.
    Will shrink keep both num_channels and spatial dimensions unchanged.
    """
    def __init__(self, in_ch: int, reluleak=0.2):
        super().__init__()
        self.block = nn.Sequential(
            conv3x3_relu_batchnorm(in_ch=in_ch, out_ch=in_ch * 2, reluleak=reluleak),
            GCU(in_ch=in_ch * 2, out_ch=in_ch),
            conv3x3_relu_batchnorm(in_ch=in_ch, out_ch=in_ch, reluleak=reluleak))

    def forward(self, x: FloatTensor) -> FloatTensor:
        return x + self.block(x)


class ConvSequence(nn.Module):
    def __init__(self, source_ch: int, nlayers: int, in_hw: int, jumpto_ch=8, direction="down"):
        super().__init__()
        (block, out_hw, out_ch) = self.build(source_ch, nlayers, in_hw, jumpto_ch, direction)
        self.block = block
        self.out_hw = out_hw
        self.out_ch = out_ch

    def build(self, source_ch: int, nlayers: int, in_hw: int, jumpto_ch=8, direction="down") -> Tuple[nn.Sequential, int, int]:
        """
        Builds our block of convolutional layers.
        Returns a Sequential module with `nlayers` layers.
        Each layer consists of a (ConvReluMaxPool, ResBlock) pair.
        Each layer will halve spatial dims, and double the number of channels.
        """
        direction = direction.lower()
        assert direction in ("down", "up")
        modules = nn.ModuleList()
        in_chs  = ((2**i) * jumpto_ch if i >= 0 else source_ch for i in range(-1, nlayers-1))   # (3, 8, 16)   given second_ch==8, nlayers==3
        out_chs = ((2**i) * jumpto_ch for i in range(nlayers))                                  # (8, 16, 32)  given second_ch==8, nlayers==3
        scalerblock = convdownblock
        # Invert all sequences and scaler block if other direction
        if direction == "up":
            in_chs, out_chs = reversed(list(out_chs)), reversed(list(in_chs))
            scalerblock = convupblock
        # Create blocks
        for (inn, out) in zip(in_chs, out_chs):
            modules.append(scalerblock(in_ch=inn, out_ch=out))
            modules.append(ResBlock(in_ch=out))
        # Create final block and compute final dimensions
        out_hw = in_hw // (2**nlayers) if (direction == "down") else in_hw * (2**nlayers)
        return (nn.Sequential(*modules), out_hw, out)

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self.block(x)
