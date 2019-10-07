import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch

class Generator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()
        self._name = 'generator_wgan'


        self.conv1 = nn.Sequential(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),\
                                    nn.InstanceNorm2d(conv_dim, affine=True), \
                                    nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        self.conv2 = nn.Sequential(nn.Conv2d(2*curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False), \
                                    nn.InstanceNorm2d(curr_dim*2, affine=True), \
                                    nn.ReLU(inplace=True))
        curr_dim = curr_dim*2
        self.conv3 = nn.Sequential(nn.Conv2d(2*curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False), \
                                    nn.InstanceNorm2d(curr_dim*2, affine=True), \
                                    nn.ReLU(inplace=True))
        curr_dim = curr_dim*2

        # Bottleneck
        layers = []
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.residual = nn.Sequential(*layers)

        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(2*curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False), \
                                    nn.InstanceNorm2d(curr_dim//2, affine=True), \
                                    nn.ReLU(inplace=True))
        curr_dim = curr_dim //2
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(2*curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False), \
                                    nn.InstanceNorm2d(curr_dim//2, affine=True), \
                                    nn.ReLU(inplace=True))
        curr_dim = curr_dim //2

        layers = []
        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Sigmoid())
        self.attetion_reg = nn.Sequential(*layers)

    def forward(self, x, c, feats = None):
        # replicate spatially and concatenate domain information
        c = c.unsqueeze(2).unsqueeze(3)
        c = c.expand(c.size(0), c.size(1), x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

        conv1_out = self.conv1(x)
        conv1_out_cat = torch.cat([conv1_out, feats['conv1_out']], dim = 1) if feats is not None \
                    else torch.cat([conv1_out, torch.randn(conv1_out.size()).cuda()], dim = 1)
        conv2_out = self.conv2(conv1_out_cat)
        conv2_out_cat = torch.cat([conv2_out, feats['conv2_out']], dim = 1) if feats is not None \
                    else torch.cat([conv2_out, torch.randn(conv2_out.size()).cuda()], dim = 1)
        conv3_out = self.conv3(conv2_out_cat)
        residual_out = self.residual(conv3_out)
        residual_out_cat = torch.cat([residual_out, feats['residual_out']], dim = 1) if feats is not None \
                        else torch.cat([residual_out, torch.randn(residual_out.size()).cuda()], dim = 1)


        deconv1_out = self.deconv1(residual_out_cat)
        deconv1_out_cat = torch.cat([deconv1_out, feats['deconv1_out']], dim = 1) if feats is not None \
                        else torch.cat([deconv1_out, torch.randn(deconv1_out.size()).cuda()], dim = 1)
        deconv2_out = self.deconv2(deconv1_out_cat)
        feats = {'conv1_out':conv1_out, \
                 'conv2_out':conv2_out, \
                 'residual_out':residual_out, \
                 'deconv1_out':deconv1_out}
        return self.img_reg(deconv2_out), self.attetion_reg(deconv2_out), feats

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
