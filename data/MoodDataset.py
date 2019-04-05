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


class MoodDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(MoodDataset, self).__init__(opt, is_for_train)
        self._name = 'MoodDataset'
        self._read_dataset_paths()

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
                print 'error reading image %s, skipping sample' % os.path.join(self._imgs_dir, sample_id)
            if real_cond is None:
                print 'error reading cond %s, skipping sample' % sample_id


        #if index % 4
        desired_cond = self._generate_random_cond(upper = 1.0, lower = 0.5 )

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

    def _read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return ids
        #return [id[:-4] for id in ids]

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.train_images_folder) if self._is_for_train else os.path.join(self._root, self._opt.test_images_folder)
        info_filepath = self._opt.affectnet_info_file
        use_ids_filepath = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file

        # read ids
        self._ids = self._read_ids(use_ids_filepath)
        self._moods = self._read_info(info_filepath)
        self._ids = list(set(self._ids).intersection(set(self._moods.keys())))
        print('#data: ', len(self._ids))

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

    def _read_info(self, file_path):
        ids = np.loadtxt(file_path, delimiter = '\n', dtype = np.str)
        ids = ids[1:]
        cols = np.array([id.split(';') for id in ids])
        names = cols[:, 0]
        names = [name.split('/')[1] for name in names]
        names = [name.split(',')[0] for name in names]

        cols = cols[:, -1]
        mood = [col.split(',')[-2:] for col in cols]

        mood_dict = dict(zip(names, mood))
        keys = set(self._ids).intersection(set(mood_dict.keys()))
        mood_dict = {k:mood_dict[k] for k in keys}
        return mood_dict

    def _get_cond_by_id(self, id):
        mood = self._get_mood_by_id(id)
        return mood

    def _get_mood_by_id(self, id):
        if id in self._moods.keys():
            cond = np.array(self._moods[id], dtype = np.float32)
            cond = np.array(cond)
            cond = (cond+1)/2
            return cond
        else:
            return None

    def _get_img_by_id(self, id):
        filepath = os.path.join(self._imgs_dir, id)
        #filepath = os.path.join(self._imgs_dir, id+'_aligned')
        #filepath = os.path.join(filepath, 'face_det_000000.bmp')
        return cv_utils.read_cv2_img(filepath), filepath

    def _generate_random_cond(self, upper = 1.0, lower = 0):
        cond = None
        while cond is None:
            rand_sample_id = self._ids[random.randint(0, self._dataset_size - 1)]
            cond = self._get_mood_by_id(rand_sample_id)
            if (abs(cond)>upper).any() or (abs(cond)<lower).any():
                cond = None
                continue
            #mood += np.random.uniform(-0.1, 0.1, mood.shape)
            cond += np.random.uniform(-0.05, 0.05, mood.shape)
	    cond = np.array(mood)

        #minV = np.amin(cond)
        #maxV = np.amax(cond)
	    #if minV != maxV:
        #        cond -= minV
        #    cond /= (maxV - minV)
        return cond
