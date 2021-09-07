
#%%

from modules import Encoder, Decoder
import torch


enc = Encoder(in_ch=3, in_hw=28, n_conv_layers=3, nz=64)
x = torch.randn(1, 3, 28, 28)
(z, mu, logvar) = enc(x)


dec = Decoder(enc)
dec(z).shape

