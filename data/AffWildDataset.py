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
import pickle


class AffWildDataset(DatasetBase):
    def __init__(self, opt, is_for_train):
        super(AffWildDataset, self).__init__(opt, is_for_train)
        self._name = 'AffWildDataset'
        self._read_dataset_paths()

    def __getitem__(self, index):
        assert (index < self._dataset_size)
        frames = None
        annotations = None
        cond = None
        target_frame = None
        while frames is None or annotations is None or cond is None or target_frame is None:
            if not self._opt.serial_batches:
                index = random.randint(0, self._dataset_size - 1)

            sample_id = self._ids[index]
            annotations, frame_ids, cond, cond_id = self._get_annotations_by_id(sample_id, self._opt.frames_cnt, self._opt.frames_rng)

            for frame_id in frame_ids:
                img, img_path = self._get_img_by_id(frame_id)

                if real_img is None:
                    print 'error reading image %s, skipping sample' % os.path.join(self._imgs_dir, sample_id)
                    frames = None

                img = self._transform(Image.fromarray(img))
                frames = img if frames == None else torch.cat([frames, img], dim=0)

            target_frame, _ = self._get_img_by_id(cond_id)
            target_frame = self._transform(Image.fromarray(target_frame))

        first_frame = frames[0].clone()
        frames = torch.squeeze(frames.view(1, -1, self._opt.image_size, self._opt.image_size))
        annotations = torch.squeeze(annotations.view(1, -1))
        sample = {'frames': frames,
                  'annotations': real_cond,
                  'desired_cond': cond,
                  'cond_id': cond_id,
                  'target_frame':target_frame,
                  'first_frame':first_frame
                  }
        return sample


    def __len__(self):
        return self._dataset_size

    def _read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        ids = [id[:-4] for id in ids]
        return ids

    def _read_dataset_paths(self):
        self._root = self._opt.data_dir
        self._imgs_dir = os.path.join(self._root, self._opt.train_images_folder) if self._is_for_train else os.path.join(self._root, self._opt.test_images_folder)
        use_ids_filepath = self._opt.train_ids_file if self._is_for_train else self._opt.test_ids_file
        self._annotations_dir = self._opt.annotations_folder

        # read ids
        self._ids = self._read_ids(use_ids_filepath)
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


    def _get_annotations_by_id(self, id, cnt, rng):
        path = os.path.join(self._opt._annotations_dir, id + '.pkl')
        data = pickle.load(path)
        start = random.randint(0, len(data) - 1)
        end = start + cnt
        random_frame = min(random.randint(end+1, end+rng), len(data)-1)

        frame_ids = data.keys()[start:end]
        annotations = [data[id] for id in frame_ids]
        cond_id = data.keys()[cond_id]
        cond = data[cond_id]
        return annotations, frame_ids, cond, cond_id



    def _get_img_by_id(self, filepath):
        return cv_utils.read_cv2_img(filepath), filepath
