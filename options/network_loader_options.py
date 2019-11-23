import argparse
import os
from utils import util
import torch

class NetworkLoaderOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--model_name', type=str, default='fully_connected', help='model name to load')
        #FULLY CONNECTED NET ARGUMENTS
        self._parser.add_argument('--train_ids_file', type=str, default='emotion_cat_urls.csv', help='file containing train ids')
        self._parser.add_argument('--test_ids_file', type=str, default='emotion_cat_urls.csv', help='file containing test ids')
        self._parser.add_argument('--aus_file', type=str, default='aus_emotion_cat.pkl', help='file containing action units')
        self._parser.add_argument('--save_folder', type=str, default='/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/fully_connected_models', help='folder for saving models')
        self._parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='directory containing images')
        self._parser.add_argument('--name', type=str, default='experiment_1', help='name of training')
        self._parser.add_argument('--labels_file', type=str, default='emotion_cat_aws.xlsx', help='file containing train ids')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--nepochs', type=int, default=4, help='number of epochs')
        self._parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self._parser.add_argument('--num_iters_validate', type = int, default = 2, help = '# batches to use when validating')
        self._parser.add_argument('--gpu_ids', type = int, default = 0, help = 'gpu ids')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--log_dir', type=str, default='/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/fully_connected_tensorboard_log', help='folder for saving models')
        self._parser.add_argument('--rule_learning', type = int, default = -1, help = 'if it is -1, it will learn from a dataset, if it is 1, it will create random aus and from will learn from lookup table')
        self._parser.add_argument('--is_train', type=int, default=1, help='in training mode use 1, else any number')

        #CONVOLUTIONAL NETWORK ARGUMENTS
        self._parser.add_argument('--data_dir', type=str, help='path to dataset')
        self._parser.add_argument('--imgs_dir', type=str, default='imgs', help='directory containing images')
        self._parser.add_argument('--is_midfeatures_used', type=int, default=-1, help='if it is 1, middle level features are used, otherwise images are used')
        self._parser.add_argument('--image_size', type=int, default=128, help='image size')
        self._parser.add_argument('--layers','--list', action='append', help='<Required> Set flag', required=True)


        #AUS TRAINING ARGUMENTS
        self._parser.add_argument('--train_images_folder', type=str, default='imgs', help='train images folder')
        self._parser.add_argument('--test_images_folder', type=str, default='imgs', help='test images folder')
        self._parser.add_argument('--training_aus_file', type=str, default='aus_openface.pkl', help='file containing samples aus')
        self._parser.add_argument('--test_aus_file', type=str, default='aus_openface.pkl', help='file containing samples aus')
        self._parser.add_argument('--dataset_mode', type=str, default='aus', help='chooses dataset to be used')
        self._parser.add_argument('--cond_nc', type=int, default=17, help='# of conditions')
        self._parser.add_argument('--n_threads_test', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--n_threads_train', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')


        #SVM ARGUMENTS
        self._parser.add_argument('--ids_file', type=str, default='emotion_cat_urls.csv', help='file containing train ids')
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()
        self.is_train = self._opt.is_train==1

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _print(self, args):
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

    def _save(self, args):
        expr_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        print(expr_dir)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt_%s.txt' % ('train' if self.is_train else 'test'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
