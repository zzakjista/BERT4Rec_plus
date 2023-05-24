from models.base import BaseModel
import torch
from torch import nn
from torch.nn import functional as F


class AEModel(BaseModel):
    def __init__(self,args):
        super().__init__(args)
        self.input_dropout = nn.Dropout(p=args.ae_dropout)
        self.encoder = nn.Sequential(
            nn.Linear(args.ae_num_items, args.ae_latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.ae_latent_dim, args.ae_num_items),
            nn.Sigmoid(),
        )
        self.encoder.apply(self.weight_init)
        self.decoder.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    @classmethod
    def code(cls):
        return 'ae'
