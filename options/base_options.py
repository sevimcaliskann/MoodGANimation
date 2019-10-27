import argparse
import os
from utils import util
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class BaseOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self._initialized = False

    def initialize(self):
        self._parser.add_argument('--data_dir', type=str, help='path to dataset')
        self._parser.add_argument('--train_ids_file', type=str, default='train_ids.csv', help='file containing train ids')
        self._parser.add_argument('--test_ids_file', type=str, default='test_ids.csv', help='file containing test ids')
        self._parser.add_argument('--train_images_folder', type=str, default='imgs', help='train images folder')
        self._parser.add_argument('--test_images_folder', type=str, default='imgs', help='test images folder')
        #self._parser.add_argument('--affectnet_info_file', type=str, default='imgs', help='file to read moods and emo from affectnet')
        self._parser.add_argument('--train_info_file', type=str, default='imgs', help='file to read moods and emo from affectnet')
        self._parser.add_argument('--test_info_file', type=str, default='imgs', help='file to read moods and emo from affectnet')
        self._parser.add_argument('--training_aus_file', type=str, default='aus_openface.pkl', help='file containing samples aus')
        self._parser.add_argument('--test_aus_file', type=str, default='aus_openface.pkl', help='file containing samples aus')
        self._parser.add_argument('--vid_frames_nums', type=str, default='vid_frames.pkl', help='file containing number of frames for each video clip)
        self._parser.add_argument('--aus_folder', type=str, default='aus', help='file containing samples aus')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--image_size', type=int, default=128, help='input image size')
        self._parser.add_argument('--frames_cnt', type=int, default=9, help='input image size')
        self._parser.add_argument('--frames_rng', type=int, default=60, help='input image size')
        self._parser.add_argument('--cond_nc', type=int, default=17, help='# of conditions')
        self._parser.add_argument('--aus_dim', type=int, default=17, help='# of action units')
        self._parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self._parser.add_argument('--name', type=str, default='experiment_1', help='name of the experiment. It decides where to store samples and models')
        self._parser.add_argument('--dataset_mode', type=str, default='aus', help='chooses dataset to be used')
        self._parser.add_argument('--model', type=str, default='ganimation', help='model to run[au_net_model]')
        self._parser.add_argument('--n_threads_test', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self._parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self._parser.add_argument('--do_saturate_mask', action="store_true", default=False, help='do use mask_fake for mask_cyc')
        self._parser.add_argument('--face_gpu_id', type=str, default='8', help='gpu for pretrained cnn model for face detection')
        self._parser.add_argument('--recurrent', type=str2bool, nargs='?', const=True, default=False, help='activate recurrent training')
        self._initialized = True

    def parse(self):
        if not self._initialized:
            self.initialize()
        self._opt = self._parser.parse_args()

        # set is train or set
        self._opt.is_train = self.is_train

        # set and check load_epoch
        self._opt.load_epoch = self._set_and_check_load_epoch(self._opt.name, self._opt.load_epoch)

        # get and set gpus
        self._get_set_gpus()

        args = vars(self._opt)

        # print in terminal args
        self._print(args)

        # save args to file
        self._save(args)

        return self._opt

    def _set_and_check_load_epoch(self, name, epoch):
        models_dir = os.path.join(self._opt.checkpoints_dir, name)
        if os.path.exists(models_dir):
            if epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % epoch
        else:
            assert epoch < 1, 'Model for epoch %i not found' % epoch
            epoch = 0

        return epoch

    def _get_set_gpus(self):
        # get gpu ids
        str_ids = self._opt.gpu_ids.split(',')
        self._opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.gpu_ids.append(id)

        str_ids = self._opt.face_gpu_id.split(',')
        self._opt.face_gpu_id = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self._opt.face_gpu_id.append(id)

        # set gpu ids
        if len(self._opt.gpu_ids) > 0:
            torch.cuda.set_device(self._opt.gpu_ids[0])

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
