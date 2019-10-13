import torch.nn as nn
import numpy as np
from .networks import NetworkBase
import torch


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], 1, -1)
        return x

class Discriminator(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6):
        super(Discriminator, self).__init__()
        self._name = 'discriminator_wgan'

        feat_layers = []
        feat_layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        feat_layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            feat_layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            feat_layers.append(nn.LeakyReLU(0.01, inplace=True))
            #if i <=repeat_num/2:
            #    self.feat_layers.append(nn.Sequential(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1), \
            #                                 nn.LeakyReLU(0.01, inplace=True)))
            #else:
            #    self.feat_layers.append(nn.Sequential(nn.Conv2d(2*curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1), \
            #                                 nn.LeakyReLU(0.01, inplace=True)))

            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        feat_layers.append(Flatten())
        self.main = nn.Sequential(*feat_layers)

        self.gru = nn.GRU(curr_dim, hidden_size=128, num_layers = 2, batch_first=True )


        self.lin1 = nn.Linear(128, 1)
        self.lin2 = nn.Linear(128, c_dim)

    def forward(self, x, hidden=None):
        h = self.main(x)
        if hidden is None:
            hidden=torch.randn(2, h.size(0), 128).cuda()
        out, hidden = self.gru(h, hidden)

        out_real = self.lin1(out.squeeze())
        out_aux = self.lin2(out.squeeze())
        return (out_real.squeeze(), out_aux.squeeze(), hidden)

    def load_from_checkpoint(self, save_dir, epoch_label):
        load_filename = 'net_epoch_%s_id_D.pth' % (epoch_label)
        load_path = os.path.join(save_dir, load_filename)
        state_dict = torch.load(load_path, map_location='cuda:0')
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)

    '''def carry_to_cuda(self):
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
        return params'''
