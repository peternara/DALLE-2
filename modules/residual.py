
from torch import nn, FloatTensor
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self, in_ch: int, res_ch: int) -> None:
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(),
            # 3x3 conv, spatial same
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=res_ch,
                kernel_size=3, 
                stride=1, 
                padding=1, 
                bias=False),
            nn.ReLU(),
            # 1v1 Conv, spatial same
            nn.Conv2d(
                in_channels=res_ch,
                out_channels=in_ch,
                kernel_size=1, 
                stride=1,
                bias=False))
        
    def forward(self, x: FloatTensor) -> FloatTensor:
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_ch: int, res_ch: int, nlayers: int):
        super(ResidualStack, self).__init__()
        self._layers = nn.ModuleList([Residual(in_ch=in_ch, res_ch=res_ch) for _ in range(nlayers)])
    
    def forward(self, x: FloatTensor) -> FloatTensor:
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)
