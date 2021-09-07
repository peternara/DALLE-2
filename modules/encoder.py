


from torch import nn, FloatTensor
from .residual import ResidualStack


class Encoder(nn.Module):
    def __init__(self, in_ch: int, nh: int, res_ch: int, n_res_layers: int):
        super().__init__()
        self._nh = nh
        self._layers = nn.Sequential(
            # Shrink spatial by 2
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=nh//2,
                kernel_size=4,
                stride=2, 
                padding=1),
            nn.ReLU(),
            # Shrink spatial by 2
            nn.Conv2d(
                in_channels=nh//2,
                out_channels=nh,
                kernel_size=4,
                stride=2, 
                padding=1),
            nn.ReLU(),
            # Keep spatial same
            nn.Conv2d(
                in_channels=nh,
                out_channels=nh,
                kernel_size=3,
                stride=1, 
                padding=1),
            # Keep spatial same
            ResidualStack(
                in_ch=nh, 
                res_ch=res_ch, 
                nlayers=n_res_layers))
    
    @property
    def nh(self) -> int:
        return self._nh

    def forward(self, x: FloatTensor) -> FloatTensor:
        return self._layers(x)
