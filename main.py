
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import FloatTensor
from torch.optim import Adam, Optimizer, optimizer
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import load_data
from modules import VQVAE, Decoder, Encoder, Quantiser
import matplotlib.pyplot as plt
from torchvision.utils import make_grid




class VQVAETrainer:
    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._tb = SummaryWriter()

    def _create_trainloader(self, batch_size: int):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ))
        ])
        trainset = datasets.MNIST('dataset/', train=True, download=True, transform=transform)
        testset = datasets.MNIST('dataset/', train=False, download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        self._loader = trainloader
        return trainloader

    def _train_step(self, batch: FloatTensor, model: VQVAE) -> FloatTensor:
        batch = batch[0].to(self._device)
        quant_loss, recon_x = model(batch)
        recon_loss = F.mse_loss(recon_x, batch)
        loss = recon_loss + quant_loss
        self._tb.add_scalar(tag="Reconstruction Loss", scalar_value=recon_loss)
        self._tb.add_scalar(tag="Quantisation Loss", scalar_value=quant_loss)
        self._tb.add_scalar(tag="Overall Loss", scalar_value=loss)
        return loss

    def _optimize_step(self, loss: FloatTensor, optimizer: Optimizer) -> None:
        loss.backward()
        optimizer.step()

    def train(self, model: VQVAE, epochs: int, batch_size: int, lr: float):
        model = model.to(self._device)
        model.train()
        optimizer = Adam(model.parameters(), lr=lr, amsgrad=False)
        trainloader = self._create_trainloader(batch_size=batch_size)
        for e in tqdm(range(epochs)):
            for batch in trainloader:
                optimizer.zero_grad()
                loss = self._train_step(batch, model)
                self._optimize_step(loss=loss, optimizer=optimizer)
            self.evaluate(e, model)
                
    def evaluate(self, epoch: int, model: VQVAE) -> None:
        model.eval()
        (originals, _) = next(iter(self._loader))
        originals = originals[:9, :, :, :].to(self._device)
        _, recon_x = model(originals)
        def show(img):
            npimg = img.numpy()
            fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig(f"screenshots/epoch_{epoch}.png")
        show(make_grid(recon_x.cpu().data, nrow=3)+0.5, )
        

def main(args: argparse.Namespace) -> None:
    model = VQVAE(
        encoder=Encoder(in_ch=args.in_ch, nh=args.nh, res_ch=args.res_ch, n_res_layers=args.n_res_layers),
        quantiser=Quantiser(num_embeddings=args.num_embeddings, embedding_dim=args.emb_dimensions),
        decoder=Decoder(in_ch=args.emb_dimensions, nh=args.nh, res_ch=args.res_ch, n_res_layers=args.n_res_layers))
    trainer = VQVAETrainer(device=args.device)
    trainer.train(model=model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    trainer.evaluate()
    model.save_weights(args.saved_weights_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dimensions
    parser.add_argument("--in_ch", type=int, default=1, help="Number of channels of the input data")
    parser.add_argument("--res_ch", type=int, default=3, help="Number of residual filter channels in encoder/decoder")
    parser.add_argument("--n_res_layers", type=int, default=2, help="Number of residual blocks to stack in encoder/decoder")
    parser.add_argument("--nh", type=int, default=128, help="Ultimate number of hidden channels (encoder final number of channels and decoder initial channels)")
    parser.add_argument("--in_h", type=int, default=28, help="Height of input data images in pixels")
    parser.add_argument("--in_w", type=int, default=28, help="Width of input data images in pixels")
    parser.add_argument("--num_embeddings", type=int, default=512, help="Number of VQVAE codebook entries")
    parser.add_argument("--emb_dimensions", type=int, default=64, help="Dimensions of each codebook entry")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs to train for")
    parser.add_argument("--batch_size", type=int, default=2048, help="Mini batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate of Optimizer")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use ('cuda' if gpu, else 'cpu')")
    parser.add_argument("--saved_weights_dir", type=str, default="saved_weights/weights.pt", help="Save path")
    (args, _) = parser.parse_known_args()
    main(args)
