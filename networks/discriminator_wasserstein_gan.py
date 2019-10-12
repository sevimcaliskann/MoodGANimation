import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch

class Discriminator(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        self._name = 'discriminator_wgan'

        self.feat_layers = []
        self.feat_layers.append(nn.Sequential(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1), \
                                         nn.LeakyReLU(0.01, inplace=True)))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            if i <=repeat_num/2:
                self.feat_layers.append(nn.Sequential(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1), \
                                             nn.LeakyReLU(0.01, inplace=True)))
            else:
                self.feat_layers.append(nn.Sequential(nn.Conv2d(2*curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1), \
                                             nn.LeakyReLU(0.01, inplace=True)))

            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=k_size, bias=False)

    def forward(self, x, feats=None):
        h = self.feat_layers[0](x)
        next_feats = dict()
        for idx in range(1, len(self.feat_layers)):
            if idx<= len(self.feat_layers)/2:
                h = self.feat_layers[idx](h)
            else:
                next_feats[idx] = h
                h_cat = torch.cat([h, feats[idx]], dim=1) if feats is not None \
                    else torch.cat([h, torch.randn(h.size()).cuda()], dim=1)
                h = self.feat_layers[idx](h_cat)

        out_real = self.conv1(h)
        out_aux = self.conv2(h)
        return (out_real.squeeze(), out_aux.squeeze(), next_feats)

    def carry_to_cuda(self):
        for layer in self.feat_layers:
            layer.cuda()
        self.conv1.cuda()
        self.conv2.cuda()

    def get_parameters(self):
        params = list(self.feat_layers[0].parameters())
        for idx in range(1, len(self.feat_layers)):
            params += list(self.feat_layers[idx].parameters())
        params += list(self.conv1.parameters())
        params += list(self.conv2.parameters())
        return params
