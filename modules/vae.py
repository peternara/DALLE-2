
#%%



from typing import Tuple
import torch
from torch import FloatTensor, nn, exp
from torch.autograd import Variable
import re

from layers import ConvSequence, convKxK


class Encoder(nn.Module):
    def __init__(self, in_ch: int, in_hw: int, n_conv_layers: int, nz: int) -> None:
        super().__init__()
        self.source_hw = in_hw
        self.source_ch = in_ch
        self.nz = nz
        self.nlayers = n_conv_layers
        # Create Convolutional sequence
        seq = ConvSequence(source_ch=in_ch, in_hw=in_hw, nlayers=n_conv_layers, direction="down")
        self.convencoder = seq
        self.flatten = nn.Flatten()
        # Save useful attributes
        self.final_hw = seq.out_hw
        self.final_ch = seq.out_ch
        self.final_flattened_dims = (seq.out_ch) * (seq.out_hw ** 2)
        # Create mu & logvar linear layers
        self.fc_mu = nn.Linear(in_features=self.final_flattened_dims, out_features=nz)
        self.fc_var = nn.Linear(in_features=self.final_flattened_dims, out_features=nz)

    def encode(self, x: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """
        Args:
            x: Image tensor, shape of (batch, NC, H, W)

        Output:
            Linear tensor, shape of (batch, NC * 16, H//4, W//4)
        """
        enc = self.flatten(self.convencoder(x))
        mu, logvar = self.fc_mu(enc), self.fc_var(enc)
        return (mu, logvar)

    def reparametrize(self, mu: FloatTensor, logvar: FloatTensor) -> FloatTensor:
        std = exp(logvar * 0.5)
        eps = Variable(torch.randn(logvar.shape))
        return mu + (eps * std)

    def forward(self, x: FloatTensor) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        (mu, logvar) = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return (z, mu, logvar)



class Decoder(nn.Module):
    def __init__(self, enc: Encoder):
        super().__init__()
        self.fc_z = nn.Linear(in_features=enc.nz, out_features=enc.final_flattened_dims)
        self.reshape = nn.Unflatten(dim=1, unflattened_size=(enc.final_ch, enc.final_hw, enc.final_hw))
        self.convup = ConvSequence(source_ch=enc.source_ch, nlayers=enc.nlayers, in_hw=enc.final_hw, direction="up")
        self.outconv = convKxK(in_channels=enc.source_ch, out_channels=enc.source_ch, in_hw=self.convup.out_hw, out_hw=enc.source_hw)

    def forward(self, z: FloatTensor) -> FloatTensor:
        x = self.fc_z(z)
        x = self.reshape(x)
        x = self.convup(x)
        return torch.sigmoid(self.outconv(x))


class VAE(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: FloatTensor) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:
        (z, mu, logvar) = self.encoder(x)
        recon_x = self.decoder(z)
        return (recon_x, mu, logvar)



class VariationalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCELoss

    def forward(self, x: FloatTensor, recon_x: FloatTensor, mu: FloatTensor, logvar: FloatTensor) -> FloatTensor:
        batch = x.shape[0]
        bceloss = self.bceloss(recon_x, x)
        kld_element = logvar + 1 - mu**2 - exp(logvar)
        kldloss = -0.5 * kld_element.sum()
        return (bceloss + kldloss) / batch

