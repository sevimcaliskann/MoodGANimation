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
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from networks.networks import NetworkBase
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from sklearn.utils.class_weight import compute_class_weight
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp


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







class FaceDataset(Dataset):

    def __init__(self, opt, is_for_train):
        self._opt = opt
        self._is_for_train = is_for_train
        ids_filepath = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        self.ids = self.read_ids(ids_filepath)
        self.get_labels(self._opt.labels_file)
        self.root_dir = self._opt.data_dir if self._opt.is_midfeatures_used==1 else self._opt.imgs_dir
        self.create_transform()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        data = None
        if self._opt.is_midfeatures_used==1:
            file_name = os.path.join(self.root_dir, self.ids[idx]+'.pkl')
            with open(file_name, 'r') as file:
                data = pickle.load(file)
                data = np.concatenate((data['ResidualBlock:2'], data['ResidualBlock:3']), axis = 1)
                data = np.squeeze(data, axis=0)
        else:
            file_name = os.path.join(self.root_dir, self.ids[idx]+'.jpg')
            data = io.imread(file_name)
        #label = self.labels[self.ids[idx]]
        label, = np.where(self.labels[self.ids[idx]]==1)
        #label = label.astype('float')


        #if self.transform:
            #data = self.transform(data)

        sample = {'data': data, 'label': label}



        return sample

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
                if(len(key)>0) and key in self.ids:
                    self.labels[key] = emo

        self.ids = list(set(self.ids).intersection(set(self.labels.keys())))


    def get_weights(self):
        y = np.zeros(len(self.ids))
        for i, id in enumerate(self.ids):
            temp = self.labels[id]
            temp, = np.where(temp == 1)
            temp = temp[0]
            y[i] = temp

        self.weights = compute_class_weight('balanced', np.arange(16), y)
        return self.weights

    def read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return [id[:-4] for id in ids]

    def create_transform(self):
        if self._is_for_train:
            transform_list = [transforms.Resize(size=(self._opt.image_size, self._opt.image_size)), transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        else:
            transform_list = [transforms.Resize(size=(self._opt.image_size, self._opt.image_size)), transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        self.transform = transforms.Compose(transform_list)



class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x



class ConvNet(NetworkBase):
    def __init__(self, opt, conv_dim=32, weights = None):
        super(ConvNet, self).__init__()
        self._opt = opt
        c_dim = 512 if self._opt.is_midfeatures_used else 3
        layers = []
        layers.append(nn.Conv2d(c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))
        layers.append(Flatten())
        layers.append(nn.Linear(conv_dim*self._opt.image_size*self._opt.image_size, 64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(64, 16))
        self.net = nn.Sequential(*layers)
        self.net.cuda()
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate,
                                             betas=[0.5, 0.999])

        self._Tensor = torch.cuda.FloatTensor if self._opt.gpu_ids else torch.Tensor
        #self.criterion = torch.nn.MSELoss().cuda()
        if weights is None:
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.criterion = torch.nn.CrossEntropyLoss(weight=weights).cuda()
        self.criterion.size_average = False
        self.data = self._Tensor(self._opt.batch_size, c_dim, self._opt.image_size, self._opt.image_size)
        #self.label = self._Tensor(self._opt.batch_size, 16).long()
        self.label = self._Tensor(self._opt.batch_size, 1).long()




    def network_set_input(self, input):
        self.data.reshape(input['data'].size()).copy_(input['data'])
        self.label.reshape(input['label'].size()).copy_(input['label'])

        #self.data = Variable(self.data.cuda(0, async=True), requires_grad = True)
        #self.label = Variable(self.label.cuda(0, async=True), requires_grad = True)


        #self.label = Variable(self.label, requires_grad = True)
        #self.label_cuda = self.label.cuda(0, async=True)

        self.data = self.data.cuda(0, async=True)
        self.label = self.label.cuda(0, async=True)

    def forward_net(self):
        pred = self.net(self.data)
        #print('pred: ', pred)
        #print('label: ', self.label)
        loss = self.criterion(pred, self.label.squeeze(dim=1))

        return loss, pred

    def optimize_parameters(self):
        loss, pred = self.forward_net()
        self.net.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss = loss.cpu().detach().item()
        return loss
        #label_grad = np.squeeze(self.label.grad.data.numpy(), axis=1)
        #return loss, label_grad



class ConvNetTrain:
    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self.initialize()
        self._opt = self._parser.parse_args()

        dataset_train = FaceDataset(self._opt, is_for_train = True)
        dataset_test = FaceDataset(self._opt, is_for_train = False)
        self._dataset_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self._opt.batch_size,
            drop_last=True)
        self._dataset_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=self._opt.batch_size,
            drop_last=True)
        self.dataset_train_size = len(dataset_train)
        self.dataset_test_size = len(dataset_test)
        print('#train images = %d' % self.dataset_train_size)
        print('#test images = %d' % self.dataset_test_size)

        self.writer = SummaryWriter()


        self._save_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if not os.path.exists(self._save_dir):
            os.makedirs(self._save_dir)
        self.net = ConvNet(self._opt, weights = torch.FloatTensor(dataset_train.get_weights()).cuda())

        self.set_and_check_load_epoch()
        if self._opt.load_epoch > 0:
            self.load()


        #self.train()
        self.eval()


    def set_and_check_load_epoch(self):
        models_dir = os.path.join(self._opt.checkpoints_dir, self._opt.name)
        if os.path.exists(models_dir):
            if self._opt.load_epoch == -1:
                load_epoch = 0
                for file in os.listdir(models_dir):
                    if file.startswith("net_epoch_"):
                        load_epoch = max(load_epoch, int(file.split('_')[2]))
                self._opt.load_epoch = load_epoch
                print('LOAD EPOCH: ', load_epoch)
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


    def initialize(self):
        self._parser.add_argument('--data_dir', type=str, help='path to dataset')
        self._parser.add_argument('--train_ids_file', type=str, default='emotion_cat_urls.csv', help='file containing train ids')
        self._parser.add_argument('--test_ids_file', type=str, default='emotion_cat_urls.csv', help='file containing test ids')
        self._parser.add_argument('--imgs_dir', type=str, default='imgs', help='directory containing images')
        self._parser.add_argument('--save_folder', type=str, default='/home/sevim/Downloads/master_thesis_study_documents/code-examples/GANimation/latent_training/dumb data', help='folder for saving models')
        self._parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='directory containing images')
        self._parser.add_argument('--name', type=str, default='experiment_1', help='name of training')
        self._parser.add_argument('--labels_file', type=str, default='emotion_cat_aws.xlsx', help='file containing train ids')
        self._parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
        self._parser.add_argument('--nepochs', type=int, default=4, help='number of epochs')
        self._parser.add_argument('--print_freq_s', type=int, default=60, help='frequency of showing training results on console')
        self._parser.add_argument('--is_midfeatures_used', type=int, default=-1, help='if it is 1, middle level features are used, otherwise images are used')
        self._parser.add_argument('--image_size', type=int, default=128, help='image size')
        self._parser.add_argument('--num_iters_validate', type = int, default = 2, help = '# batches to use when validating')
        self._parser.add_argument('--gpu_ids', type = int, default = 0, help = 'gpu ids')
        self._parser.add_argument('--load_epoch', type=int, default=-1, help='which epoch to load? set to -1 to use latest cached model')


    def train(self):
        self._iters_per_epoch = self.dataset_train_size/ self._opt.batch_size
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
            self.net.network_set_input(train_batch)
            self.loss = self.net.optimize_parameters()
            #self.loss, self.label_grad = self.net.optimize_parameters()
            #print('label grad: ', label_grad)

            # update epoch info
            epoch_iter += self._opt.batch_size

            # display terminal
            if do_print_terminal:
                self.display_terminal(iter_start_time, i_epoch, i_train_batch)
                self._last_print_time = time.time()






    def save(self, label):
        # save networks
        self.save_network(self.net, self._opt.name, label)
        self.save_optimizer(self.net.optimizer, self._opt.name, label)

    def load(self):
        load_epoch = self._opt.load_epoch
        self.load_network(self.net, self._opt.name, load_epoch)
        self.load_optimizer(self.net.optimizer, self._opt.name, load_epoch)


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
            self.net.network_set_input(val_batch)
            loss, _ = self.net.forward_net()
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

        #print('PARAM KEYS: ', self.net.net.state_dict().keys())

        #for name, param in self.net.net.state_dict().items():
        #    if param.grad == None:
        #        continue
        #    print('params: ', name)
        #self.writer.add_histogram('label_grad', self.label_grad, i_epoch*self._iters_per_epoch + i_train_batch)

    def save_optimizer(self, optimizer, optimizer_label, epoch_label):
        save_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, optimizer, optimizer_label, epoch_label):
        load_filename = 'opt_epoch_%s_id_%s.pth' % (epoch_label, optimizer_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path

        optimizer.load_state_dict(torch.load(load_path, map_location='cuda:0'))
        print 'loaded optimizer: %s' % load_path

    def save_network(self, network, network_label, epoch_label):
        save_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self._save_dir, save_filename)
        torch.save(network.state_dict(), save_path)
        print 'saved net: %s' % save_path

    def load_network(self, network, network_label, epoch_label):
        load_filename = 'net_epoch_%s_id_%s.pth' % (epoch_label, network_label)
        load_path = os.path.join(self._save_dir, load_filename)
        assert os.path.exists(
            load_path), 'Weights file not found. Have you trained a model!? We are not providing one' % load_path
        #network = torch.nn.DataParallel(network)
        network.load_state_dict(torch.load(load_path, map_location='cuda:0'))
        print 'loaded net: %s' % load_path



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


    def eval(self):
        self.set_and_check_load_epoch()
        if self._opt.load_epoch > 0:
            self.load()

        accuracy = 0
        y_predicts = []
        y_targets = []
        x = []
        val_errors = OrderedDict()
        for i_val_batch, val_batch in enumerate(self._dataset_test):

            # evaluate model
            self.net.network_set_input(val_batch)
            target = np.around(val_batch['label'].cpu().detach().numpy()).astype('int32')
            #target, = np.where(target==1)
            loss, pred = self.net.forward_net()
            if 'val_loss' in val_errors:
                val_errors['val_loss'] += loss
            else:
                val_errors['val_loss'] = loss
            #pred = np.around(pred.cpu().detach().numpy()).astype('int32')
            pred = np.around(pred.cpu().detach().numpy())
            #pred, = np.where(pred == np.max(pred))

            for y_t, y_p in zip(target, pred):
                y_p, = np.where(y_p == np.max(y_p))
                y_p = y_p[0]
                if y_t==y_p:
                    accuracy = accuracy +1
                y_predicts.append(y_p)
                y_targets.append(y_t)

        y_predicts = np.array(y_predicts)
        y_targets = np.squeeze(np.array(y_targets), axis=1)
        print('Accuracy: ', float(accuracy) /self.dataset_test_size)
        conf_mat = confusion_matrix(y_targets, y_predicts)
        print('Confusion matrix: ', conf_mat)
        plot_consuion_matrix(y_targets, y_predicts, self._opt.save_folder)
        plot_roc_curves(y_targets, y_predicts, np.arange(16), self._opt.save_folder)



        #fpr, tpr, thresholds = roc_curve(y_targets, y_predicts, pos_label=1)
        #auc = roc_auc_score(y_targets, y_predicts)
        #fig, ax = plt.subplots()
        #ax.plot(fpr, tpr)
        #ax.plot([0, 1], [0, 1], color='navy', linestyle='--', label='random')
        #plt.title('AUC: {auc}')
        #ax.set_xlabel('False positive rate')
        #ax.set_ylabel('True positive rate')
        #fig.savefig('plots.pdf')   # save the figure to file
        #plt.close(fig)



if __name__ == "__main__":
    ConvNetTrain()
