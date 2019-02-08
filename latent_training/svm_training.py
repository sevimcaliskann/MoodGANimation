import time
import os
import numpy as np
from sklearn import linear_model
from collections import OrderedDict
import argparse
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix
import cv2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import log_loss
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import pyplot as plt

def plot_consuion_matrix(y_target, y_pred, save_dir):
    labels = ['Angrily disgusted', 'Angrily surprised', 'Angry', 'Appalled', 'Awed', 'Disgusted', 'Fearful', 'Fearfully angry', 'Fearfully surprised', 'Happily disgusted', 'Happily surprised', 'Happy', 'Sad', 'Sadly angry', 'Sadly disgusted', 	'Surprised']
    cm = confusion_matrix(y_target, y_pred, np.arange(16))
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels, fontsize=8)
    ax.set_yticklabels([''] + labels, fontsize=8)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(save_dir, 'Confusion_Matrix.png'))

def plot_roc_curves(y_target, y_predict, cls, save_dir):
    labels = ['Angrily disgusted', 'Angrily surprised', 'Angry', 'Appalled', 'Awed', 'Disgusted', 'Fearful', 'Fearfully angry', 'Fearfully surprised', 'Happily disgusted', 'Happily surprised', 'Happy', 'Sad', 'Sadly angry', 'Sadly disgusted', 	'Surprised']
    y_target = label_binarize(y_target, classes=cls)
    y_predict = label_binarize(y_predict, classes=cls)
    n_classes = y_target.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_target[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_target.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure(figsize=(12, 10))
    lw = 2
    #plt.plot(fpr[2], tpr[2], color='darkorange',
    #         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    for i, label in zip(range(n_classes), labels):
        plt.plot(fpr[i], tpr[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(label, roc_auc[i]))
    #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'ROC_Curves.png'))   # save the figure to file


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

        self.weights = self.get_weights()
        self.weights = dict(zip(np.arange(16), self.weights))
        self.clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, class_weight=self.weights, loss = 'modified_huber', n_jobs = 8, warm_start=True)

        self.writer = SummaryWriter(log_dir='svm_record')

        self.train()
        self.eval()


    def initialize(self):
        self._parser.add_argument('--data_dir', type=str, help='path to dataset')
        self._parser.add_argument('--ids_file', type=str, default='emotion_cat_urls.csv', help='file containing train ids')
        self._parser.add_argument('--imgs_dir', type=str, default='imgs', help='directory containing images')
        self._parser.add_argument('--labels_file', type=str, default='emotion_cat_aws.xlsx', help='file containing train ids')
        self._parser.add_argument('--save_folder', type=str, default='/home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/latent_training/svm_models', help='folder for saving models')
        self._parser.add_argument('--model_name', type=str, default='experiment_1', help='model_name')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--nepochs', type=int, default=4, help='number of epochs')
        self._parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self._parser.add_argument('--is_midfeatures_used', type=int, default=-1, help='if it is 1, middle level features are used, otherwise images are used')

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
        self.writer.close()
        filename = os.path.join(self._opt.save_folder, self._opt.model_name + '.sav')
        pickle.dump(self.clf, open(filename, 'wb'))

    def train_epoch(self, i_epoch):
        print('Iteration per epoch: ', self._iters_per_epoch)
        for i_train_batch in range(self._iters_per_epoch):
            x,y = self.get_batch(i_train_batch)
            b, c, w, h = x.shape
            x = x.reshape(b, c*w*h)
            y = [list(row).index(1) for row in y]
            iter_start_time = time.time()
            do_print_terminal = time.time() - self._last_print_time > self._opt.print_freq_s

            # train model
            #self.clf.warm_start = True
            #self.clf.n_jobs = 8
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
            message = '(target: %d, prediction: %d) ' % (label, y_predict)
            print(message)
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


    def eval(self):
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
            message = '(target: %d, prediction: %d) ' % (label, y_predict)
            print(message)
            if label == y_predict:
                accuracy = accuracy + 1
        print('Accuracy: ', float(accuracy) / len(self.test_ids))
        conf_mat = confusion_matrix(y_targets, y_predicts)
        print('Confusion matrix: ', conf_mat)
        plot_consuion_matrix(y_targets, y_predicts, self._opt.save_folder)
        plot_roc_curves(y_targets, y_predicts, np.arange(16), self._opt.save_folder)


        #fpr, tpr, thresholds = roc_curve(y_targets, y_predicts, pos_label=1)
        #auc = roc_auc_score(y_targets, y_predicts)
        #fig, ax = plt.subplots()
        #ax.plot(fpr, tpr)
        #ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
        #plt.title(f'AUC: {auc}')
        #ax.set_xlabel('False positive rate')
        #ax.set_ylabel('True positive rate')
        #fig.savefig(os.path.join(self._opt.save_folder, 'plots.pdf'))   # save the figure to file
        #plt.close(fig)



if __name__ == "__main__":
    SvmTrain()
