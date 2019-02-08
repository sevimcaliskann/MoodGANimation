import os.path
import torchvision.transforms as transforms
from data.dataset import DatasetBase
#from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import pickle
from utils import cv_utils
#import face_recognition
#import dlib
#from skimage import io


class AusDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(AusDataset, self).__init__(opt, is_for_train)
        self._name = 'AusDataset'

        self._read_dataset_paths()

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
        while real_img is None or real_cond is None:
            # if sample randomly: overwrite index
            if not self._opt.serial_batches:
                index = random.randint(0, self._dataset_size - 1)

            # get sample data
            sample_id = self._ids[index]

            real_img, real_img_path = self._get_img_by_id(sample_id)
            real_cond = self._get_cond_by_id(sample_id)

            if real_img is None:
                print 'error reading image %s, skipping sample' % os.path.join(self._imgs_dir, sample_id+'.jpg')
            if real_cond is None:
                print 'error reading aus %s, skipping sample' % sample_id

        desired_cond = self._generate_random_cond()

        # transform data
        img = self._transform(Image.fromarray(real_img))

        # pack data
        sample = {'real_img': img,
                  'real_cond': real_cond,
                  'desired_cond': desired_cond,
                  'sample_id': sample_id,
                  'real_img_path': real_img_path
                  }

        # print (time.time() - start_time)

        return sample

    def __len__(self):
        return self._dataset_size

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.train_images_folder) if self._is_for_train else os.path.join(self._root, self._opt.test_images_folder)
        conds_filepath = self._opt.training_aus_file if self._is_for_train else self._opt.test_aus_file

        # read ids
        use_ids_filepath = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        self._ids = self._read_ids(use_ids_filepath)

        # read aus
        self._conds = self._read_conds(conds_filepath)
        self._ids = list(set(self._ids).intersection(set(self._conds.keys())))

        # dataset size
        self._dataset_size = len(self._ids)

    '''def _read_dataset_paths_multi(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.train_images_folder) if self._is_for_train else os.path.join(self._root, self._opt.test_images_folder)

        # read ids
        use_ids_filepath = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        self._ids = self._read_ids(use_ids_filepath)

        # read aus
        conds_folderpath = self._opt.aus_folder
        folders = next(os.walk(conds_folderpath))[1]
	    self._conds=dict()
        for folder in folders:
            folder_path = os.path.join(conds_folderpath, folder)
            files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pkl')]
            for f in files:
                conds = self._read_conds(os.path.join(folder_path, f))
                self._conds.update(conds)

        self._ids = list(set(self._ids).intersection(set(self._conds.keys())))

        # dataset size
        self._dataset_size = len(self._ids)'''

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

    def _get_cond_by_id(self, id):
        if id in self._conds:
            #cond = self._conds[id]
            #if cond.shape[0]==0:
                #return None
            #minV = np.amin(cond)
            #maxV = np.amax(cond)
            #if minV!=maxV:
            #    cond -= minV
            #    cond /= (maxV - minV)
            cond = np.array(self._conds[id], dtype = np.float32)/5.0
            return cond
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

        #minV = np.amin(cond)
        #maxV = np.amax(cond)
	    #if minV != maxV:
        #        cond -= minV
        #    cond /= (maxV - minV)
        return cond
