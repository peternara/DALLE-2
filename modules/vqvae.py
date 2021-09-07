
from typing import Tuple
from torch import nn, FloatTensor
from dataclasses import dataclass
import torch

from .encoder import Encoder
from .decoder import Decoder
from .quantiser import Quantiser



class VQVAE(nn.Module):
    def __init__(self, encoder: Encoder, quantiser: Quantiser, decoder: Decoder) -> None:
        super().__init__()
        self._encoder = encoder
        self._conv1x1 = nn.Conv2d(
            in_channels=encoder.nh,
            out_channels=quantiser.embedding_dim,
            kernel_size=1,
            stride=1)
        self._quantiser = quantiser
        self._decoder = decoder

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> "VQVAE":
        self.load_state_dict(torch.load(path))

    def forward(self, x: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """
        Inputs:
            x: FloatTensor of shape (batch, nc, h, w)
        """
        z = self._encoder(x)      # (batch, nh, h//4, w//4)
        quant_loss, quantised = self._quantiser(self._conv1x1(z))
        recon_x = self._decoder(quantised)
        return quant_loss, recon_x
