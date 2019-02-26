import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import argparse
from torch.autograd import Variable
from collections import OrderedDict
import time
import os
import numpy as np
from options.network_loader_options import NetworkLoaderOptions
from .aus_training import AUsTrainer
from latent_training.conv_net_train import FaceDataset
from latent_training.fully_connected_train import FullyConnectedNet
from torch.utils.data import Dataset
import utils.test_utils as tutils
from sklearn.metrics import f1_score



def eval(aus_net, fully_connected_net, dataset_test, dataset_test_size):
    accuracy = 0
    y_predicts = []
    y_targets = []
    for i_val_batch, val_batch in enumerate(dataset_test):

        # evaluate model
        aus_net.set_input(val_batch)
        target = np.around(val_batch['label'].cpu().detach().numpy()).astype('int32')
        pred, _ = aus_net.forward()
        pred = pred.cpu().detach().numpy()
        pred = torch.cuda.FloatTensor(pred)
        label = [0]
        label = torch.cuda.LongTensor(label)

        data = {'data':pred, 'label':label}
        fully_connected_net.network_set_input(data)
        #target, = np.where(target==1)
        _, emo_pred = fully_connected_net.forward_net()

        emo_pred = np.around(emo_pred.cpu().detach().numpy())
        #pred, = np.where(pred == np.max(pred))

        for y_t, y_p in zip(target, emo_pred):
            y_p, = np.where(y_p == np.max(y_p))
            y_p = y_p[0]
            if y_t==y_p:
                accuracy = accuracy +1
            y_predicts.append(y_p)
            y_targets.append(y_t)

    y_predicts = np.array(y_predicts)
    y_targets = np.array(y_targets)
    accuracy = float(accuracy) /dataset_test_size
    f1 = f1_score(y_targets, y_predicts, average='macro')
    print('Accuracy: ', accuracy)
    print('F1 Score: ', f1)
    tutils.save_confusion_matrix(y_targets, y_predicts, '/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/aus_training')
    tutils.plot_roc_curves(y_targets, y_predicts, np.arange(16), '/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/aus_training')
    return accuracy


def load_network(network, network_label, epoch_label, dir):
    load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
    load_path = os.path.join(dir, load_filename)
    assert os.path.exists(
        load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path
    #network = torch.nn.DataParallel(network)
    network.load_state_dict(torch.load(load_path))
    print 'loaded net: %s' % load_path


def load_optimizer(optimizer, optimizer_label, epoch_label, dir):
    load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
    load_path = os.path.join(dir, load_filename)
    assert os.path.exists(
        load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

    optimizer.load_state_dict(torch.load(load_path))
    print 'loaded optimizer: %s' % load_path


if __name__ == "__main__":
    opt = NetworkLoaderOptions().parse()

    fully_connected_net = FullyConnectedNet(opt)
    #fully_connected_net._opt.name = 'fully_connect_weightless_cross_rule_learning'
    load_network(fully_connected_net, fully_connected_net._opt.name, 72, '/srv/glusterfs/csevim/datasets/emotionet/emotion_cat/checkpoints/fully_connect_weightless_cross_rule_learning')
    load_optimizer(fully_connected_net.optimizer, fully_connected_net._opt.name, 72, '/srv/glusterfs/csevim/datasets/emotionet/emotion_cat/checkpoints/fully_connect_weightless_cross_rule_learning')

    dataset_test = FaceDataset(opt, is_for_train = False)
    dataset_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=opt.batch_size,
        drop_last=True)
    dataset_test_size = len(dataset_test)
    print('#test images = %d' % dataset_test_size)

    opt.name='aus_train'
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model = AUsTrainer(save_dir=save_dir)
    model.load_network(18)
    model.load_optimizer(18)

    accuracies = eval(model, fully_connected_net, dataset_test, dataset_test_size)
    print('accuracies: ', accuracies)
    #plot_accuracies(accuracies, '/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/aus_training')
