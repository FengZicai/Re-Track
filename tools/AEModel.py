import torch
import torch.nn as nn

import tools.PCEncoderDecoder as PCEncoderDecoder
from tools.DenseAE import DPCEncoderDecoder

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.bneck_size = encoder.bneck_size

    def forward(self, X):
        return self.decoder(self.encoder(X))

    def encode(self, X):
        return self.encoder(X)

    def decode(self, X):
        return self.decoder(X)


class PCAutoEncoder(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''
    def __init__(self, bneck_size=128):
        self.input_size = 2048
        encoder = PCEncoderDecoder.Encoder(
            bneck_size=bneck_size, input_size=self.input_size)
        decoder = PCEncoderDecoder.Decoder(bneck_size=bneck_size)
        super().__init__(encoder, decoder)


class DPAutoEncoder(nn.Module):
    '''
    The AutoEncoder is redesigned, which is uesd to extract and utilize more features for point-clouds.
    '''
    def __init__(self, bneck_size=128):
        super(DPAutoEncoder, self).__init__()
        self.bneck_size = bneck_size
        self.input_size = 2048
        self.DAE = DPCEncoderDecoder(bneck_size=bneck_size, input_size=self.input_size)

    def forward(self, X):
        output, _ = self.DAE(X)
        return output

    def encode(self, X):
        _, encoded = self.DAE(X)
        return encoded
