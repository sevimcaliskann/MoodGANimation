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
from sklearn.metrics import f1_score


class SvmTrain:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self.initialize()
        self._opt = self._parser.parse_args()
        self.ids = self.read_ids(self._opt.ids_file)
        self.get_labels(self._opt.labels_file)
        self.weights = self.get_weights()
        self.weights = dict(zip(np.arange(16), self.weights))


        if self._opt.input_mode=='au' and self._opt.randomize_aus==1:
            self.aus_dict, self.labels = tutils.create_aus_lookup()
        elif self._opt.input_mode == 'au':
            self.read_aus()
        self.writer = SummaryWriter(log_dir=self._opt.log_dir)


        if self._opt.is_train==1 and self._opt.kfold<3:
            self.separate_train_test()
            print('#train images = %d' % len(self.train_ids))
            print('#test images = %d' % len(self.test_ids))
            if self._opt.randomize_aus==1:
                self.clf = linear_model.SGDClassifier(max_iter=100, tol=1e-3, loss = 'modified_huber', n_jobs = 8, warm_start=True)
            else:
                self.clf = linear_model.SGDClassifier(max_iter=100, tol=1e-3, class_weight=self.weights, loss = 'modified_huber', n_jobs = 8, warm_start=True)
            self.set_and_check_load_epoch()
            self.train()
            self.eval()
        elif self._opt.is_train==1 and self._opt.kfold>=3:
            self._opt.nepochs /= self._opt.kfold
            self.fold_accuracies = []
            for i in range(self._opt.kfold):
                self.fold_index = i
                self._opt.load_epoch = -1 #####CLOSE THIS ONE LATER ON!!!
                self.set_and_check_load_epoch()
                print('Loaded epoch for fold %s is: %s' % (self.fold_index, self._opt.load_epoch))
                if self._opt.load_epoch == 0:
                    if self._opt.randomize_aus==1:
                        self.clf = linear_model.SGDClassifier(max_iter=100, tol=1e-3, loss = 'modified_huber', n_jobs = 8, warm_start=True)
                    else:
                        self.clf = linear_model.SGDClassifier(max_iter=100, tol=1e-3, class_weight=self.weights, loss = 'modified_huber', n_jobs = 8, warm_start=True)

                    self.separate_train_test(fold_index = i)
                    print('#train images = %d' % len(self.train_ids))
                    print('#test images = %d' % len(self.test_ids))
                else:
                    self.load_classifier(self._opt.load_epoch)
                    self.separate_train_test(fold_index = i)
                    print('#train images = %d' % len(self.train_ids))
                    print('#test images = %d' % len(self.test_ids))
                if self._opt.is_train==1:
                    self.train()
                self.eval()
                avg_acc = np.mean(np.array(self.fold_accuracies))
                print('Average Accuracy: ', avg_acc)


    def initialize(self):
        self._parser.add_argument('--data_dir', type=str, help='path to dataset')
        self._parser.add_argument('--ids_file', type=str, default='emotion_cat_urls.csv', help='file containing train ids')
        self._parser.add_argument('--imgs_dir', type=str, default='imgs', help='directory containing images')
        self._parser.add_argument('--labels_file', type=str, default='emotion_cat_aws.xlsx', help='file containing train ids')
        self._parser.add_argument('--save_folder', type=str, default='/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/svm_models', help='folder for saving models')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--nepochs', type=int, default=4, help='number of epochs')
        self._parser.add_argument('--kfold', type=int, default=10, help='cross-validation k')
        self._parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')
        self._parser.add_argument('--log_dir', type=str, default='/scratch_net/zinc/csevim/apps/repos/GANimation/latent_training/svm_tensorboard_log', help='folder for saving models')
        self._parser.add_argument('--input_mode', type=str, default='res', help='either res for residual blocks, image for images and au for action units to assign as input')
        self._parser.add_argument('--aus_file', type=str, default='/srv/glusterfs/csevim/datasets/emotionet/emotion_cat/aus_emotion_cat.pkl', help='path to dictionary which stores action units')
        self._parser.add_argument('--is_train', type=int, default=1, help='in training mode use 1, else any number')
        self._parser.add_argument('--randomize_aus', type=int, default=-1, help='for rule learning, to randomize au intensities set it to 1, otherwise anything else')

    def read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return [id[:-4] for id in ids]

    def separate_train_test(self, train_ratio = 0.9, fold_index = 0):
        if self._opt.kfold<3:
            shuffles = np.random.permutation(len(self.ids))
            train_num = int(len(self.ids)*train_ratio) + 1
            train_indices = shuffles[:train_num]
            test_indices = shuffles[train_num:]
            self.train_ids = [self.ids[index] for index in train_indices]
            self.test_ids = [self.ids[index] for index in test_indices]
        else:
            indices = np.arange(len(self.ids))
            test_number = int(round(float(len(self.ids))/self._opt.kfold))
            test_indices = np.arange(test_number*fold_index, min(len(self.ids), test_number*(fold_index+1)))
            train_indices = np.delete(indices, test_indices)
            self.train_ids = [self.ids[index] for index in train_indices]
            self.test_ids = [self.ids[index] for index in test_indices]



    def read_aus(self):
        with open(self._opt.aus_file, 'rb') as f:
            self.aus = pickle.load(f)




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

        for i_epoch in range(self._opt.load_epoch + 1, self._opt.nepochs+1):
            epoch_start_time = time.time()

            # train epoch
            self.train_epoch(i_epoch)

            # print epoch info
            time_epoch = time.time() - epoch_start_time
            print('End of epoch %d / %d \t Time Taken: %d sec (%d min or %d h)' %
                  (i_epoch, self._opt.nepochs, time_epoch,
                   time_epoch / 60, time_epoch / 3600))

            if i_epoch %100 == 0 and self._opt.kfold<3:
                self.save_classifier(i_epoch)
            elif i_epoch%10 == 0 and self._opt.kfold>=3:
                self.save_classifier(i_epoch)
        #self.writer.close() ################OPEN THIS ONE WHENEVER YOU DO NOT USE CROSSFOLD


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
                loss = log_loss(y, y_pred, labels = np.arange(16))
                self.writer.add_scalar('train_loss', loss, i_epoch*self._iters_per_epoch + i_train_batch)
                self.display_terminal(iter_start_time, i_epoch, i_train_batch)
                self._last_print_time = time.time()

    def read_sample(self, id):
        if self._opt.input_mode=='res':
            block = None
            with open(os.path.join(self._opt.data_dir, id + '.pkl'), 'r') as file:
                data = pickle.load(file)
                res1 = data['ResidualBlock:2']
                res2 = data['ResidualBlock:3']
                block = np.append(res1, res2, axis=1)
            return block, self.labels[id]
        elif self._opt.input_mode=='au' and self._opt.randomize_aus==1:
            sample, _ = tutils.create_sample(self.aus_dict, self.labels)
            aus = sample['data']
            aus = np.expand_dims(np.expand_dims(np.expand_dims(aus, axis=0), axis=0), axis=0)
            label = np.ones((16,))
            label *= -1
            label[sample['label']] = 1
            return aus, label
        elif self._opt.input_mode=='au':
            aus = self.aus[id]
            aus = np.expand_dims(np.expand_dims(np.expand_dims(aus, axis=0), axis=0), axis=0)
            return aus, self.labels[id]
        else:
            img = cv2.imread(os.path.join(self._opt.imgs_dir, id + '.jpg'), -1)
            img = cv2.resize(img, (128, 128))
            img = np.rollaxis(img, 2)
            img = np.expand_dims(img, axis = 0)
            return img, self.labels[id]


    def get_batch(self, i_epoch):
        indices = np.arange(i_epoch*self._opt.batch_size, (i_epoch+1)*self._opt.batch_size)
        ids = [self.train_ids[index] for index in indices]
        #print('ids: ', ids)
        if self._opt.input_mode =='res':
            x = np.empty(shape=(0, 512, 32, 32))
        elif self._opt.input_mode =='au':
            x = np.empty(shape=(0, 1, 1, 17))
        else:
            x = np.empty(shape=(0, 3, 128, 128))

        y = np.empty(shape=(0, 16))
        for id in ids:
            sample, label = self.read_sample(id)
            x = np.append(x, sample, axis = 0)
            label = np.expand_dims(label, axis=0)
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
        accuracy = 0
        for id in ids:
            sample, _ = self.read_sample(id)
            label = self.labels[id]
            b, c, w, h = sample.shape
            sample = sample.reshape(b, c*w*h)
            label = list(label).index(1)

            y_predict = self.clf.predict(sample)
            diff = abs(label - y_predict)
            if diff==0:
                accuracy = accuracy+1
            error = error + np.sum(diff)


        error = float(error) / self._opt.batch_size
        accuracy = float(accuracy)/self._opt.batch_size

        loss_dict = OrderedDict([#('g_fake', self._loss_g_fake.detach()),
                                 ('val_acc', accuracy),
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
        if self._opt.kfold<3:
            save_filename = 'svm_epoch_%s.sav' % (epoch_label)
        else:
            save_filename = 'svm_fold_%s_epoch_%s.sav' % (self.fold_index, epoch_label)
        save_path = os.path.join(self._opt.save_folder, save_filename)
        pickle.dump(self.clf, open(save_path, 'wb'))
        print('Saved classifier to: ', save_path)

    def load_classifier(self, epoch_label):
        if self._opt.kfold<3:
            load_filename = 'svm_epoch_%s.sav' % (epoch_label)
        else:
            load_filename = 'svm_fold_%s_epoch_%s.sav' % (self.fold_index, epoch_label)
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
                    if self._opt.kfold<3 and file.startswith("svm_epoch_"):
                        load_epoch = max(load_epoch, int(os.path.splitext(file)[0].split('_')[2]))
                    elif file.startswith('svm_fold_%s' % (self.fold_index)):
                        load_epoch = max(load_epoch, int(os.path.splitext(file)[0].split('_')[4]))
                self._opt.load_epoch = load_epoch
            else:
                found = False
                for file in os.listdir(models_dir):
                    if self._opt.kfold<3 and file.startswith("svm_epoch_"):
                        found = int(file.split('_')[2]) == self._opt.load_epoch
                    elif file.startswith('svm_fold_%s' % (self._opt.kfold)):
                        found = int(file.split('_')[4]) == self._opt.load_epoch
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
            if self._opt.input_mode=='au':
                sample, label = self.read_sample(id)
            else:
                sample, label = self.read_sample(id)
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
        f1 = f1_score(y_targets, y_predicts, average='macro')
        if self._opt.kfold>=3:
            print('Fold number: ', self.fold_index)
            self.fold_accuracies.append(accuracy)
        print('Accuracy: ', accuracy)
        print('F1 Score: ', f1)
        if load_last_epoch:
            conf_mat = confusion_matrix(y_targets, y_predicts)
            print('Confusion matrix: ', conf_mat)
            conf_mat_save_name = 'Confusion_Matrix_fold_%s.png' % (self.fold_index) if self._opt.kfold>=3 else None
            roc_curves_save_name = 'ROC_Curves_fold_%s.png' % (self.fold_index) if self._opt.kfold>=3 else None
            tutils.save_confusion_matrix(y_targets, y_predicts, self._opt.save_folder, conf_mat_save_name)
            tutils.plot_roc_curves(y_targets, y_predicts, np.arange(16), self._opt.save_folder, roc_curves_save_name)
        return accuracy



if __name__ == "__main__":
    SvmTrain()
