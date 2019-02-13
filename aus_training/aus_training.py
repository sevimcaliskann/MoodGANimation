import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from data.custom_dataset_data_loader import CustomDatasetDataLoader
import numpy as np
from networks.networks import NetworkBase
from tensorboardX import SummaryWriter
import argparse
from torch.autograd import Variable
from collections import OrderedDict
import time
import os
import numpy as np

class AUsTrainer(NetworkBase):
    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=6, batch_size=16, lambda_gp = 10, save_dir = '.'):
        super(AUsTrainer, self).__init__()
        self._name = 'discriminator_wgan'
        self.c_dim = c_dim

        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        layers.append(nn.Conv2d(curr_dim, self.c_dim, kernel_size=k_size, bias=False))
        self.net = nn.Sequential(*layers)
        self.net.cuda()
        self.learning_rate=1e-4
        self._optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate,
                                             betas=[0.5, 0.999])
        self._criterion = torch.nn.MSELoss().cuda()
        self._Tensor = torch.cuda.FloatTensor
        self._input_img = self._Tensor(batch_size, 3, image_size, image_size)
        self._aus = self._Tensor(batch_size, self.c_dim)
        self._lambda_gp = lambda_gp
        self._save_dir = save_dir


    def forward(self, img=None):
        if img is None:
            out_aux = self.net(self._input_img).squeeze()
        else:
            out_aux = self.net(img).squeeze()
        loss = self._criterion(self._aus, out_aux)
        return out_aux, loss


    def set_input(self, input):
        self._input_img.resize_(input['real_img'].size()).copy_(input['real_img'])
        self._aus.resize_(input['real_cond'].size()).copy_(input['real_cond'])


        #if len(self._gpu_ids) > 0:
        self._input_img = self._input_img.cuda()
        self._aus = self._aus.cuda()

    def _gradient_penalty(self, out_aux):
        # interpolate sample
        #alpha = torch.rand(self._input_img.size(0), self.c_dim).cuda().expand_as(self._aus)
        interpolated = Variable(self._input_img, requires_grad=True)

        #self._input_img.copy_(interpolated)
        interpolated_out, _ = self.forward(interpolated)

        # compute gradients
        grad = torch.autograd.grad(outputs=interpolated_out,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_out.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        self._loss_d_gp = torch.mean((grad_l2norm - 1) ** 2) * self._lambda_gp

        return self._loss_d_gp

    def optimize_parameters(self):
        out_aux, loss = self.forward()
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        loss_gp= self._gradient_penalty(out_aux)
        self._optimizer.zero_grad()
        loss_gp.backward()
        self._optimizer.step()
        return loss + loss_gp


    def save_optimizer(self, epoch_label):
        save_filename = 'opt_epoch_%s.pth' % (epoch_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(self._optimizer.state_dict(), save_path)

    def load_optimizer(self, epoch_label):
        load_filename = 'opt_epoch_%s.pth' % (epoch_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        self._optimizer.load_state_dict(torch.load(load_path))
        print 'loaded optimizer: %s' % load_path

    def save_network(self, epoch_label):
        save_filename = 'net_epoch_%s.pth' % (epoch_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(self.net.state_dict(), save_path)
        print 'saved net: %s' % save_path

    def load_network(self, epoch_label):
        load_filename = 'net_epoch_%s.pth' % (epoch_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path
        #network = torch.nn.DataParallel(network)
        self.net.load_state_dict(torch.load(load_path))
        print 'loaded net: %s' % load_path

class AUsTrain:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self.initialize()
        self._opt = self._parser.parse_args()

        data_loader_train = CustomDatasetDataLoader(self._opt, is_for_train=True)
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)

        self._dataset_train = data_loader_train.load_data()
        self._dataset_test = data_loader_test.load_data()

        self._dataset_train_size = len(data_loader_train)
        self._dataset_test_size = len(data_loader_test)
        print('#train images = %d' % self._dataset_train_size)
        print('#test images = %d' % self._dataset_test_size)
        print('TRAIN IMAGES FOLDER = %s' % data_loader_train._dataset._imgs_dir)
        print('TEST IMAGES FOLDER = %s' % data_loader_test._dataset._imgs_dir)

        self.writer = SummaryWriter(log_dir = self._opt.log_dir)
        self.net = AUsTrainer(c_dim=self._opt.cond_nc, save_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name))

	checkpoints_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
	if not os.path.exists(checkpoints_dir):
	    os.makedirs(checkpoints_dir)        
	self.set_and_check_load_epoch()
        if self._opt.load_epoch > 0:
            self.load()


        if self._opt.is_train==1:
            self.train()
        self.eval()


    def initialize(self):
        self._parser.add_argument('--data_dir', type=str, help='path to dataset')
        self._parser.add_argument('--train_images_folder', type=str, default='imgs', help='train images folder')
        self._parser.add_argument('--test_images_folder', type=str, default='imgs', help='test images folder')
        self._parser.add_argument('--train_ids_file', type=str, default='train_ids.csv', help='file containing train ids')
        self._parser.add_argument('--test_ids_file', type=str, default='test_ids.csv', help='file containing test ids')
        self._parser.add_argument('--training_aus_file', type=str, default='aus_openface.pkl', help='file containing samples aus')
        self._parser.add_argument('--test_aus_file', type=str, default='aus_openface.pkl', help='file containing samples aus')
        self._parser.add_argument('--dataset_mode', type=str, default='aus', help='chooses dataset to be used')
        self._parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='directory containing images')
        self._parser.add_argument('--name', type=str, default='experiment_1', help='name of training')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--image_size', type=int, default=128, help='input image size')
        self._parser.add_argument('--nepochs', type=int, default=4, help='number of epochs')
        self._parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self._parser.add_argument('--num_iters_validate', type = int, default = 2, help = '# batches to use when validating')
        self._parser.add_argument('--gpu_ids', type = int, default = 0, help = 'gpu ids')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--log_dir', type=str, default='/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/aus_training', help='folder for saving models')
        self._parser.add_argument('--cond_nc', type=int, default=17, help='# of conditions')
        self._parser.add_argument('--n_threads_test', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--n_threads_train', default=1, type=int, help='# threads for loading data')
        self._parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self._parser.add_argument('--is_train', type=int, default=1, help='in training mode use 1, else any number')


    def set_and_check_load_epoch(self, is_train = True):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1 or not is_train:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0




    def train(self):
        self._iters_per_epoch = self._dataset_train_size/ self._opt.batch_size
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()



        for i_epoch in range(self._opt.load_epoch + 1, self._opt.nepochs+1):
            epoch_start_time = time.time()

            # train epoch
            self.train_epoch(i_epoch)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs, time_epoch,
                   time_epoch / 60, time_epoch / 3600))
            self.save(i_epoch)
        self.writer.close()



    def train_epoch(self, i_epoch):
        epoch_iter = 0
        print('Iteration per epoch: ', self._iters_per_epoch)
        for i_train_batch, train_batch in enumerate(self._dataset_train):
            iter_start_time = time.time()

            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s
            self.net.set_input(train_batch)
            self.loss = self.net.optimize_parameters()

            # update epoch info
            epoch_iter += self._opt.batch_size

            # display terminal
            if do_print_terminal:
                self.display_terminal(iter_start_time, i_epoch, i_train_batch)
                self._last_print_time = time.time()






    def save(self, label):
        # save networks
        self.net.save_network(label)
        self.net.save_optimizer(label)

    def load(self):
        load_epoch = self._opt.load_epoch
        self.net.load_network(load_epoch)
        self.net.load_optimizer(load_epoch)


    def display_terminal(self, iter_start_time, i_epoch, i_train_batch):
        train_errors = OrderedDict()
        train_errors['train_loss'] = self.loss

        self.writer.add_scalar('data/train_loss', self.loss, i_epoch*self._iters_per_epoch + i_train_batch)

        t = (time.time() - iter_start_time) / self._opt.batch_size
        self.print_current_train_errors(i_epoch, i_train_batch, self._iters_per_epoch, train_errors, t)

        val_start_time = time.time()

        # set model to eval
        self.net.eval()

        # evaluate self._opt.num_iters_validate epochs
        val_errors = OrderedDict()
        for i_val_batch, val_batch in enumerate(self._dataset_test):
            if i_val_batch == self._opt.num_iters_validate:
                break

            # evaluate model
            self.net.set_input(val_batch)
            _, loss = self.net.forward()
            loss = loss.cpu().detach().item()

            # store current batch errors
            if 'val_loss' in val_errors:
                val_errors['val_loss'] += loss
            else:
                val_errors['val_loss'] = loss


        # normalize errors
        for k in val_errors.iterkeys():
            val_errors[k] /= self._opt.num_iters_validate

        self.writer.add_scalar('data/val_loss', val_errors['val_loss'], i_epoch*self._iters_per_epoch + i_train_batch)

        # visualize
        t = (time.time() - val_start_time)
        self.print_current_validate_errors(i_epoch, val_errors, t)

        # set model back to train
        self.net.train()



    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (epoch: %d, it: %d/%d, t/smpl: %.3fs) ' % (log_time, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        print(message)

    def print_current_validate_errors(self, epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (V, epoch: %d, time_to_val: %ds) ' % (log_time, epoch, t)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)
        print(message)


    def eval(self, load_last_epoch=True):
        self.set_and_check_load_epoch(is_train = not load_last_epoch)
        if self._opt.load_epoch > 0:
            self.load()

        accuracy = 0
        y_predicts = []
        y_targets = []
        x = []
        val_errors = OrderedDict()
        for i_val_batch, val_batch in enumerate(self._dataset_test):

            # evaluate model
            self.net.set_input(val_batch)
            target = val_batch['real_cond'].cpu().detach().numpy()
            #target, = np.where(target==1)
            pred, loss = self.net.forward()
            if 'val_loss' in val_errors:
                val_errors['val_loss'] += loss
            else:
                val_errors['val_loss'] = loss
            #pred = np.around(pred.cpu().detach().numpy()).astype('int32')
            pred = pred.cpu().detach().numpy()
            #pred, = np.where(pred == np.max(pred))

            accuracy = (accuracy + 1) if (abs(target-pred)<=0.01).all() else accuracy

        y_predicts = np.array(y_predicts)
        y_targets = np.array(y_targets)
        accuracy = float(accuracy) /self._dataset_test_size
        print('Accuracy: ', accuracy)
        return accuracy



if __name__ == "__main__":
    AUsTrain()
