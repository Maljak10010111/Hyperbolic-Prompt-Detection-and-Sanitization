import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim = 768, rep_dim = 32): # representation dimension = 32, this is the suggested value for the textual data
        super(MLP, self).__init__()
        self.rep_dim = rep_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256, bias=False),
            nn.ReLU(),
            nn.Linear(256, 128, bias=False),
            nn.ReLU(),
            nn.Linear(128, rep_dim, bias=False)
        )

    def forward(self, x):
        return self.net(x)


class MLP_Autoencoder(nn.Module):
    def __init__(self, input_dim=768, rep_dim=32):
        super(MLP_Autoencoder, self).__init__()
        self.rep_dim = rep_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, rep_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(rep_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out

