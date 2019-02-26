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
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from tqdm import tqdm

#import plotly.plotly as py
#import plotly.tools as tls



def plot_accuracies(accuracy, errors, save_dir):
    y = accuracy.values()
    N = len(y)
    x = np.arange(N) + 1
    width = 1/1.5
    fig = plt.figure(figsize=(12, 10))
    ax =  fig.add_subplot(111)
    ax.bar(x, y, yerr=errors, align='center', alpha=0.5, color='blue', capsize=10)
    ax.set_title('Mean AU intensity errors')
    ax.set_xlabel('Action Units')
    ax.set_ylabel('Mean Error')
    ax.set_xticks(x)
    ax.set_xticklabels(accuracy.keys())
    plt.savefig(os.path.join(save_dir, 'AUs_Intensity_Errors.png'))



def eval(net, dataset_test, dataset_test_size):
    aus_mapping = {0: 'AU1', \
                   1: 'AU2', \
                   2: 'AU4', \
                   3: 'AU5', \
                   4: 'AU6', \
                   5: 'AU7', \
                   6: 'AU9', \
                   7: 'AU10', \
                   8: 'AU12', \
                   9: 'AU14', \
                   10: 'AU15', \
                   11: 'AU17', \
                   12: 'AU20', \
                   13: 'AU23', \
                   14: 'AU25', \
                   15: 'AU26', \
                   16: 'AU45'}
    accuracy = dict()
    #y_predicts = []
    #y_targets = []
    x = []
    for i_val_batch, val_batch in tqdm(enumerate(dataset_test)):

        # evaluate model
        net.set_input(val_batch)
        target = val_batch['real_cond'].cpu().detach().numpy()
        pred, _ = net.forward()
        pred = pred.cpu().detach().numpy()
        diff = abs(pred - target)
        for index, y in np.ndenumerate(diff):
            key = aus_mapping[index[1]]
            if key in accuracy:
                accuracy[key].append(y)
            else:
                accuracy[key] = []
                accuracy[key].append(y)
    stds = []
    for key in accuracy.keys():
        stds.append(np.std(accuracy[key]))
        accuracy[key] = np.mean(np.array(accuracy[key]))

    print('Accuracy: ', accuracy)
    return accuracy, stds


if __name__ == "__main__":
    opt = NetworkLoaderOptions().parse()

    data_loader_test = CustomDatasetDataLoader(opt, is_for_train=False)
    dataset_test = data_loader_test.load_data()

    dataset_test_size = len(data_loader_test)
    print('#test images = %d' % dataset_test_size)
    print('TEST IMAGES FOLDER = %s' % data_loader_test._dataset._imgs_dir)

    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    model = AUsTrainer(save_dir=save_dir)
    model.load_network(18)
    model.load_optimizer(18)
    accuracies, stds = eval(model, dataset_test, dataset_test_size)
    plot_accuracies(accuracies, stds, '/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/aus_training')
