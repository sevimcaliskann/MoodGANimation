import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
#from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import pickle
from utils import cv_utils
from utils import test_utils as tutils
#import face_recognition
#import dlib
#from skimage import io


class AusDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(AusDataset, self).__init__(opt, is_for_train)
        self._name = 'AusDataset'

        self._read_dataset_paths()
        self._aus_dict, self._labels = tutils.create_aus_lookup()

        # read dataset
        #if self._opt.aus_file=='-':
        #    self._read_dataset_paths_multi()
        #else:
        #    self._read_dataset_paths()

        #dlib.cuda.set_device(self._opt.face_gpu_id[0])
        #self.cnn_face_detector = dlib.cnn_face_detection_model_v1(self._opt.face_detection_model)
        #print("dlib cuda device: ", dlib.cuda.get_device())
        #self.cnn_face_detector(io.imread('/scratch_net/zinc/csevim/apps/repos/GANimation/face_crop_model/face.png'), 1)

    def __getitem__(self, index):
        assert (index < self._dataset_size)

        # start_time = time.time()
        real_img = None
        real_cond = None
        real_emo = None
        while real_img is None or real_cond is None or real_emo is None:
            # if sample randomly: overwrite index
            if not self._opt.serial_batches:
                index = random.randint(0, self._dataset_size - 1)

            # get sample data
            sample_id = self._ids[index]

            real_img, real_img_path = self._get_img_by_id(sample_id)
            real_cond = self._get_cond_by_id(sample_id)
            real_emo = self._get_emo_by_id(sample_id)

            if real_img is None:
                print 'error reading image %s, skipping sample' % os.path.join(self._imgs_dir, sample_id+'.jpg')
            if real_cond is None:
                print 'error reading aus %s, skipping sample' % sample_id
            if real_emo is None:
                print 'error reading emo %s, skipping sample' % sample_id


        real_emo = np.array(real_emo, dtype=np.int32)
        #one_hot_emo = np.zeros((self._opt.batch_size,11))
        #one_hot_emo[(np.arange(self._opt.batch_size), np.array(real_emo))] = 1
	one_hot_emo = np.zeros((11,))
        one_hot_emo[int(real_emo)] = 1
        real_emo = one_hot_emo

        desired_cond = self._generate_random_cond()
        #desired_emo = np.zeros((self._opt.batch_size,))
        desired_emo, _ = self._get_emo_from_cond(desired_cond)
            
        #one_hot_emo = np.zeros((self._opt.batch_size,16))
        #one_hot_emo[(np.arange(self._opt.batch_size), np.array(desired_emo))] = 1
	one_hot_emo = np.zeros((16,))
        one_hot_emo[int(desired_emo)] = 1
        desired_emo = one_hot_emo

        # transform data
        img = self._transform(Image.fromarray(real_img))

        # pack data
        sample = {'real_img': img,
                  'real_cond': real_cond,
                  'real_emo':real_emo,
                  'desired_emo': desired_emo,
                  'desired_cond': desired_cond,
                  'sample_id': sample_id,
                  'real_img_path': real_img_path
                  }

        # print (time.time() - start_time)

        return sample

    def _get_emo_from_cond(self, desired_cond):
        real_emo, name = tutils.get_emotion_label_from_aus(self._aus_dict, self._labels, desired_cond)
        return real_emo, name


    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.train_images_folder) if self._is_for_train else os.path.join(self._root, self._opt.test_images_folder)
        conds_filepath = self._opt.training_aus_file if self._is_for_train else self._opt.test_aus_file
        emos_filepath = self._opt.emo_training_file if self._is_for_train else self._opt.emo_test_file

        # read ids
        use_ids_filepath = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        self._ids = self._read_ids(use_ids_filepath)

        # read aus
        self._conds = self._read_conds(conds_filepath)
        self._emos = self._read_emos(emos_filepath)
        print('#emotions coming from data: ', len(self._emos.keys()))
        self._ids = list(set(self._ids).intersection(set(self._conds.keys())))
        self._ids = list(set(self._ids).intersection(set(self._emos.keys())))

        # dataset size
        self._dataset_size = len(self._ids)

    def _create_transform(self):
        if self._is_for_train:
            transform_list = [transforms.Resize(size=(self._opt.image_size, self._opt.image_size)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        else:
            transform_list = [transforms.Resize(size=(self._opt.image_size, self._opt.image_size)),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                   std=[0.5, 0.5, 0.5]),
                              ]
        self._transform = transforms.Compose(transform_list)

    def _read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return [id[:-4] for id in ids]

    def _read_conds(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def _read_emos(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\n', dtype=np.str)
        cols = np.array([id.split('\t') for id in ids])
        labels = np.array(cols[:, 1], dtype = np.int32)
        names = cols[:, 0]
        emos = dict()
        for i in range(len(names)):
            emos[names[i]] = labels[i]
        return emos

    def _get_cond_by_id(self, id):
        if id in self._conds:
            cond = np.array(self._conds[id], dtype = np.float32)/5.0
            return cond
        else:
            return None

    def _get_emo_by_id(self, id):
        if id in self._emos:
            emo = np.array(self._emos[id], dtype = np.float32)
            return emo
        else:
            return None

    def _get_img_by_id(self, id):
        filepath = os.path.join(self._imgs_dir, id+'.jpg')
        #filepath = os.path.join(self._imgs_dir, id+'_aligned')
        #filepath = os.path.join(filepath, 'face_det_000000.bmp')
        return cv_utils.read_cv2_img(filepath), filepath

    def _generate_random_cond(self):
        cond = None
        while cond is None:
            rand_sample_id = self._ids[random.randint(0, self._dataset_size - 1)]
            cond = self._get_cond_by_id(rand_sample_id)
            cond += np.random.uniform(-0.1, 0.1, cond.shape)
	    cond[cond<0] = 0

        #minV = np.amin(cond)
        #maxV = np.amax(cond)
	    #if minV != maxV:
        #        cond -= minV
        #    cond /= (maxV - minV)
        return cond
