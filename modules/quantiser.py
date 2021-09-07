
#%%

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import FloatTensor, LongTensor, nn
import torch.nn.functional as F



class Quantiser(nn.Module):
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