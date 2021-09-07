import torch
from torch import FloatTensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from typing import Tuple
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))


data_variance = np.var(training_data.data / 255.0)




class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment: float = 0.25) -> None:
        super().__init__()
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._commitment = commitment
        self._embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    @property
    def embedding_dim(self) -> int:
        return self._embedding.embedding_dim

    def forward(self, inputs: FloatTensor) -> Tuple[FloatTensor, FloatTensor]:
        """
        Inputs:
            shape of (batch, channels, height, width)
            
        Outpus:
            loss: FloatTensor, shape of (0,)
            quantised: FloatTensor, shape of (batch, channels, height, width)
        """
        (b, nc, h, w) = inputs.shape
        emb_matrix = self._embedding.weight
        num_embeddings, embedding_dim = emb_matrix.shape
        assert nc == embedding_dim, f"Input dimension {nc} must match embedding dimension {embedding_dim}"
    
        inputs: FloatTensor = inputs.permute(0, 2, 3, 1).contiguous()   # (batch, h, w, nc)
        flat_inputs = inputs.view(b*h*w, nc)                            # (batch*h*w, nc)

        # Calculate distances
        emb_matrix = self._embedding.weight                             # (num_embeddings, nc)
        distances = (torch.sum(flat_inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_inputs, self._embedding.weight.t()))
                                                                        # Final shape = (batch*h*w, num_embeddings)
        # Perform Encoding
        encoding_indices = distances.argmin(dim=1).unsqueeze(1)         # (batch*h*w, num_embeddings) => (batch*h*w,) => (batch*h*w, 1)
        encodings = torch.zeros((b*h*w, num_embeddings), device=inputs.device)
        encodings.scatter_(dim=1, index=encoding_indices, value=1)      # (batch*h*w, num_embeddings)
        
        # Quantise, unflatten
        quantised = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        
        # Get loss
        q_latent_loss = F.mse_loss(quantised, inputs.detach())
        e_latent_loss = F.mse_loss(quantised.detach(), inputs)
        loss = q_latent_loss + (self._commitment * e_latent_loss)
        
        # Convert inputs into quantised nicely
        quantised = inputs + (quantised - inputs).detach()

        return loss, quantised.permute(0, 3, 1, 2).contiguous()
    

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
        
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_ch: int, res_ch: int, nlayers: int):
        super().__init__()
        self._layers = nn.ModuleList([Residual(in_ch, res_ch) for _ in range(nlayers)])
    
    def forward(self, x):
        for layer in self._layers:
            x = layer(x)
        return F.relu(x)
    
    

class Encoder(nn.Module):
    def __init__(self, in_ch: int, nh: int, n_res_layers: int, res_ch: int):
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
                nh,
                res_ch,
                n_res_layers))
    
    @property
    def nh(self) -> int:
        return self._nh

    def forward(self, x):
        return self._layers(x)

    

class Decoder(nn.Module):
    def __init__(self, in_ch: int, nh: int, n_res_layers: int, res_ch: int):
        super().__init__()
        self._layers = nn.Sequential(
            # Keep spatial dims unchanged
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=nh,
                kernel_size=3, 
                stride=1, 
                padding=1),
            # Keep spatial dims unchanged
            ResidualStack(
                nh,
                res_ch,
                n_res_layers),
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

    def forward(self, x):
        return self._layers(x)
    
    
    
batch_size = 256
num_training_updates = 15000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-3

training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)

validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)


class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, None
    
model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

model.train()
train_res_recon_error = []
train_res_perplexity = []

for i in range(num_training_updates):
    (data, _) = next(iter(training_loader))[:9, :, :, :]
    data = data.to(device)
    optimizer.zero_grad()

    vq_loss, data_recon, perplexity = model(data)
    print('vq_loss', vq_loss.item())
    recon_error = F.mse_loss(data_recon, data) / data_variance
    loss = recon_error + vq_loss
    loss.backward()

    optimizer.step()
    
    train_res_recon_error.append(recon_error.item())

    print('recon_error', recon_error.item())
