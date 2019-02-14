import time
import os
import numpy as np
from collections import OrderedDict
import argparse
import pandas as pd
import pickle
import cv2
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from networks.networks import NetworkBase
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from options.network_loader_options import NetworkLoaderOptions


class NetworkLoader:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        model = None

        if model_name == 'conv_net':
            from latent_training import conv_net_train
            model = conv_net_train.ConvNet(*args, **kwargs)
        elif model_name == 'fully_connected':
            from latent_training import fully_connected_train
            model = fully_connected_train.FullyConnectedNet(*args, **kwargs)
        elif model_name=='aus_trainer':
            from aus_training import aus_training
            model = aus_training.AUsTrainer(*args, **kwargs)
        elif model_name == 'svm':
            from latent_training import svm_training
            model = svm_training.SvmTrain(*args, **kwargs)
        elif model_name == 'ganimation':
            from .ganimation import GANimation
            model = GANimation(*args, **kwargs)
        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model_name)
        return model


if __name__ == "__main__":
    opt = NetworkLoaderOptions().parse()
    network_loader = NetworkLoader()
    model = network_loader.get_by_name(opt.model_name, opt)
    #Write one method for each classifier for single pass, later on
    # you are gonna use for getting result from the network and cascade to next network
