
from torch import nn, FloatTensor
from .residual import ResidualStack


class Decoder(nn.Module):
    def __init__(self, in_ch: int, nh: int, res_ch: int, n_res_layers: int):
        super().__init__()
        self._layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=nh,
                kernel_size=3, 
                stride=1, 
                padding=1),
            ResidualStack(
                in_ch=nh,
                res_ch=res_ch,
                nlayers=n_res_layers),
            # Double spatial dim
            nn.ConvTranspose2d(
                in_channels=nh, 
                out_channels=nh//2,
                kernel_size=4,
                stride=2, 
                padding=1),
            nn.ReLU(),
            # Double Spatial dim
            nn.ConvTranspose2d(
                in_channels=nh//2, 
                out_channels=3,
                kernel_size=4, 
                stride=2, 
                padding=1))
        

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self._layers(x)
