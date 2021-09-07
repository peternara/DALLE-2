
import unittest

import torch

from data import load_data
from modules import (VQVAE, Decoder, Encoder, QuantisedEncoding, Quantiser,
                     VQVAEOutput)

# General
BATCH = 8
IN_CH = 3
RES_CH = 32
N_RES_LAYERS = 4
NH = 256
IN_H = 32
IN_W = 32

# Quantiser
EMB_DIMENSIONS = 64
NUM_EMBEDDINGS = 20


class TestEncoder(unittest.TestCase):
    def test_forward_backward(self) -> None:
        model = Encoder(in_ch=IN_CH, nh=NH, res_ch=RES_CH, n_res_layers=N_RES_LAYERS)
        x = torch.rand(BATCH, IN_CH, IN_H, IN_W)
        out = model(x)
        self.assertTrue(out.shape == (BATCH, NH, IN_H//4, IN_W//4))
        out.sum().backward()


class TestQuantiser(unittest.TestCase):
    def test_forward_backward(self) -> None:
        model = Quantiser(num_embeddings=NUM_EMBEDDINGS, embedding_dim=NH)
        x = torch.rand(BATCH, NH, IN_H//4, IN_W//4)
        quantised: QuantisedEncoding = model(x)
        self.assertTrue(isinstance(quantised, QuantisedEncoding))
        self.assertTrue(quantised.quantised.shape == x.shape)
        quantised.loss.backward()


class TestDecoder(unittest.TestCase):
    def test_forward_backward(self) -> None:
        model = Decoder(in_ch=NH, nh=NH, res_ch=RES_CH, n_res_layers=N_RES_LAYERS)
        x = torch.rand(BATCH, NH, IN_H//4, IN_W//4)
        out = model(x)
        self.assertTrue(out.shape == (BATCH, IN_CH, IN_H, IN_W))


class TestVQVAE(unittest.TestCase):
    def test_forward_backward(self) -> None:
        encoder = Encoder(in_ch=IN_CH, nh=NH, res_ch=RES_CH, n_res_layers=N_RES_LAYERS)
        quantiser = Quantiser(num_embeddings=NUM_EMBEDDINGS, embedding_dim=EMB_DIMENSIONS)
        decoder = Decoder(in_ch=EMB_DIMENSIONS, nh=NH, res_ch=RES_CH, n_res_layers=N_RES_LAYERS)
        vqvae = VQVAE(encoder, quantiser, decoder)
        x = torch.rand(BATCH, IN_CH, IN_H, IN_W)
        out: VQVAEOutput = vqvae(x)


class TestLoadingData(unittest.TestCase):
    def test_loading_data(self) -> None:
        (train, valid) = load_data()
        self.assertTrue(train.data.shape == (50000, 32, 32, 3))
        self.assertTrue(valid.data.shape == (10000, 32, 32, 3))


if __name__ == "__main__":
    unittest.main()
