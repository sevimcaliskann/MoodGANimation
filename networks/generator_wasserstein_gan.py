import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch
from .convgru import ConvGRU

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(1, x.size()[0], -1)
        return x

class Deflatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(1), -1, 4, 4)
        return x

class Generator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=8, c_dim=5, repeat_num=5):
        super(Generator, self).__init__()
        self._name = 'generator_wgan'

        self.first_conv = nn.Sequential(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False), \
                                        nn.InstanceNorm2d(conv_dim, affine=True), \
                                        nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        self.encode1 = nn.Sequential(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False), \
                                     nn.InstanceNorm2d(curr_dim*2, affine=True), \
                                     nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2

        self.encode2 = nn.Sequential(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False), \
                                     nn.InstanceNorm2d(curr_dim*2, affine=True), \
                                     nn.ReLU(inplace=True))
        curr_dim = curr_dim * 2
        ## feature map sizes are 32x32

        # Bottleneck
        #layers.append(Flatten())
        self.gru = ConvGRU(input_size=curr_dim, hidden_sizes=curr_dim,
                  kernel_sizes=7, n_layers=6)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(2*curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False), \
                                     nn.InstanceNorm2d(curr_dim//2, affine=True), \
                                     nn.ReLU(inplace=True))
        curr_dim = curr_dim // 2
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2*curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False), \
                                     nn.InstanceNorm2d(curr_dim//2, affine=True), \
                                     nn.ReLU(inplace=True))
        curr_dim = curr_dim // 2

        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.attetion_reg = nn.Sequential(*layers)


    def forward(self, x, c, hidden = None):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        first = self.first_conv(x)
        encoded1 = self.encode1(first)
        encoded2 = self.encode2(encoded1)
        if hidden is None:
            out = self.gru(encoded2)
        else:
            out = self.gru(encoded2, hidden)


        decoded1 = self.deconv1(torch.cat([out[-1], encoded2], dim=1))
        decoded2 = self.deconv2(torch.cat([decoded1, encoded1], dim=1))

        return self.img_reg(decoded2), self.attetion_reg(decoded2), out

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)
