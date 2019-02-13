import time
import os
import numpy as np
from sklearn import linear_model
from collections import OrderedDict
import argparse
import pandas as pd
import cPickle as pickle
from sklearn.metrics import confusion_matrix
import cv2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss
from tensorboardX import SummaryWriter
import utils.test_utils as tutils


class SvmTrain:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self.initialize()
        self._opt = self._parser.parse_args()
        self.ids = self.read_ids(self._opt.ids_file)
        self.get_labels(self._opt.labels_file)
        self.separate_train_test()
        print('#train images = %d' % len(self.train_ids))
        print('#test images = %d' % len(self.test_ids))

        self.set_and_check_load_epoch()

        if(self._opt.load_epoch==0):
            self.weights = self.get_weights()
            self.weights = dict(zip(np.arange(16), self.weights))
            self.clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, class_weight=self.weights, loss = 'modified_huber', n_jobs = 8, warm_start=True)
        else:
            self.load_classifier(self._opt.load_epoch)

        self.writer = SummaryWriter(log_dir=self._opt.log_dir)

        if self._opt.is_train==1:
            self.train()
        self.eval()


    def initialize(self):
        self._parser.add_argument('--data_dir', type=str, help='path to dataset')
        self._parser.add_argument('--ids_file', type=str, default='emotion_cat_urls.csv', help='file containing train ids')
        self._parser.add_argument('--imgs_dir', type=str, default='imgs', help='directory containing images')
        self._parser.add_argument('--labels_file', type=str, default='emotion_cat_aws.xlsx', help='file containing train ids')
        self._parser.add_argument('--save_folder', type=str, default='/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/svm_models', help='folder for saving models')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--nepochs', type=int, default=4, help='number of epochs')
        self._parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self._parser.add_argument('--is_midfeatures_used', type=int, default=-1, help='if it is 1, middle level features are used, otherwise images are used')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--log_dir', type=str, default='/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/svm_tensorboard_log', help='folder for saving models')
        self._parser.add_argument('--is_train', type=int, default=1, help='in training mode use 1, else any number')

    def read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return [id[:-4] for id in ids]

    def separate_train_test(self, train_ratio = 0.9):
        shuffles = np.random.permutation(len(self.ids))
        train_num = int(len(self.ids)*train_ratio) + 1
        train_indices = shuffles[:train_num]
        test_indices = shuffles[train_num:]
        self.train_ids = [self.ids[index] for index in train_indices]
        self.test_ids = [self.ids[index] for index in test_indices]

    def get_labels(self, filepath):
        self.labels = dict()
        xl_file = pd.ExcelFile(filepath)
        for sheet_name in xl_file.sheet_names:
            chunk = xl_file.parse(sheet_name)
            urls = np.array(chunk[chunk.columns[0]])
            emos = np.array(chunk[chunk.columns[2:]])
            #for count in range(len(urls)):
            for url, emo in zip(urls, emos):
                key = url.split('/')[-1][:-4]
                if(len(key)>0):
                    self.labels[key] = emo

    def get_weights(self):
        y = np.zeros(len(self.ids))
        for i, id in enumerate(self.ids):
            temp = self.labels[id]
            temp, = np.where(temp == 1)
            temp = temp[0]
            y[i] = temp

        weights = compute_class_weight('balanced', np.arange(16), y)
        return weights

    def train(self):
        self._iters_per_epoch = len(self.train_ids) / self._opt.batch_size
        self._last_display_time = None
        self._last_save_latest_time = None
        self._last_print_time = time.time()

        for i_epoch in range(self._opt.nepochs):
            epoch_start_time = time.time()

            # train epoch
            self.train_epoch(i_epoch)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs, time_epoch,
                   time_epoch / 60, time_epoch / 3600))

            if i_epoch %100 == 0:
                self.save_classifier(i_epoch)
        self.writer.close()


    def train_epoch(self, i_epoch):
        print('Iteration per epoch: ', self._iters_per_epoch)
        for i_train_batch in range(self._iters_per_epoch):
            x,y = self.get_batch(i_train_batch)
            b, c, w, h = x.shape
            x = x.reshape(b, c*w*h)
            y = [list(row).index(1) for row in y]
            iter_start_time = time.time()
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s
            self.clf.partial_fit(x,y, np.arange(16))

            # display terminal
            if do_print_terminal:
                y_pred = self.clf.predict_proba(x)
                #print('y_pred: ', y_pred)
                #print('y_true: ', y)
                loss = log_loss(y, y_pred, labels = np.arange(16))
                self.writer.add_scalar('train_loss', loss, i_epoch*self._iters_per_epoch + i_train_batch)
                self.display_terminal(iter_start_time, i_epoch, i_train_batch)
                self._last_print_time = time.time()

    def read_sample(self, id):
        if self._opt.is_midfeatures_used==1:
            block = None
            with open(os.path.join(self._opt.data_dir, id + '.pkl'), 'r') as file:
                data = pickle.load(file)
                res1 = data['ResidualBlock:2']
                res2 = data['ResidualBlock:3']
                block = np.append(res1, res2, axis=1)
            return block
        else:
            img = cv2.imread(os.path.join(self._opt.imgs_dir, id + '.jpg'), -1)
            img = cv2.resize(img, (128, 128))
            img = np.rollaxis(img, 2)
            img = np.expand_dims(img, axis = 0)
            return img


    def get_batch(self, i_epoch):
        indices = np.arange(i_epoch*self._opt.batch_size, (i_epoch+1)*self._opt.batch_size)
        ids = [self.train_ids[index] for index in indices]
        #print('ids: ', ids)
        x = np.empty(shape=(0, 512, 32, 32)) if self._opt.is_midfeatures_used==1 else np.empty(shape=(0, 3, 128, 128))
        y = np.empty(shape=(0, 16))
        for id in ids:
            sample = self.read_sample(id)
            x = np.append(x, sample, axis = 0)
            label = np.expand_dims(self.labels[id], axis=0)
            y = np.append(y, label, axis=0)
        return x,y

    def display_terminal(self, iter_start_time, i_epoch, i_train_batch):
        errors = self.get_current_errors()

        t = (time.time() - iter_start_time) / self._opt.batch_size
        self.print_current_train_errors(i_epoch, i_train_batch, self._iters_per_epoch, errors, t)

    def get_current_errors(self):
        indices = np.random.permutation(len(self.test_ids))
        indices = indices[:5]
        ids = [self.test_ids[index] for index in indices]
        error = 0
        for id in ids:
            sample = self.read_sample(id)
            label = self.labels[id]
            b, c, w, h = sample.shape
            sample = sample.reshape(b, c*w*h)
            label = list(label).index(1)

            y_predict = self.clf.predict(sample)
            diff = abs(label - y_predict)
            error = error + np.sum(diff)

        error = error / self._opt.batch_size

        loss_dict = OrderedDict([#('g_fake', self._loss_g_fake.detach()),
                                 #('g_cond', self._loss_g_cond.data[0]),
                                 ('val_error', error)])



        return loss_dict

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, t):
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (epoch: %d, it: %d/%d, t/smpl: %.3fs) ' % (log_time, epoch, i, iters_per_epoch, t)
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        print(message)
        self.writer.add_scalar('svm_val_error', errors['val_error'], epoch*iters_per_epoch + i)


    def save_classifier(self, epoch_label):
        save_filename = 'svm_epoch_%s.sav' % (epoch_label)
        save_path = os.path.join(self._opt.save_folder, save_filename)
        pickle.dump(self.clf, open(save_path, 'wb'))
        print('Saved classifier to: ', save_path)

    def load_classifier(self, epoch_label):
        load_filename = 'svm_epoch_%s.sav' % (epoch_label)
        load_path = os.path.join(self._opt.save_folder, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        self.clf = pickle.load(open(load_path, 'rb'))
        print 'loaded optimizer: %s' % load_path



    def set_and_check_load_epoch(self, is_train = True):
        models_dir = self._opt.save_folder
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1 or not is_train:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("svm_epoch_"):
                        load_epoch = max(load_epoch, int(os.path.splitext(file)[0].split('_')[2]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if file.startswith("svm_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                        if found: break
                assert found, 'Model for epoch %i not found' % self._opt.load_epoch
        else:
            assert self._opt.load_epoch < 1, 'Model for epoch %i not found' % self._opt.load_epoch
            self._opt.load_epoch = 0


    def eval(self, load_last_epoch = True):
        self.set_and_check_load_epoch(is_train=not load_last_epoch)
        self.load_classifier(self._opt.load_epoch)
        accuracy = 0
        y_predicts = []
        y_targets = []
        for id in self.test_ids:
            sample = self.read_sample(id)
            label = self.labels[id]
            b, c, w, h = sample.shape
            sample = sample.reshape(b, c*w*h)
            label = list(label).index(1)
            y_targets.append(label)

            y_predict = self.clf.predict(sample)
            y_predicts.append(y_predict)
            if label == y_predict:
                accuracy = accuracy + 1

        accuracy = float(accuracy) / len(self.test_ids)
        print('Accuracy: ', accuracy)
        if load_last_epoch:
            conf_mat = confusion_matrix(y_targets, y_predicts)
            print('Confusion matrix: ', conf_mat)
            tutils.save_confusion_matrix(y_targets, y_predicts, self._opt.save_folder)
            tutils.plot_roc_curves(y_targets, y_predicts, np.arange(16), self._opt.save_folder)
        return accuracy



if __name__ == "__main__":
    SvmTrain()
