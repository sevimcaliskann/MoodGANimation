import time
from models.models import ModelsFactory
import os
import torch
import numpy as np
import argparse
from options.test_options import TestOptions
import torchvision.transforms as transforms
from utils import face_utils
from utils import cv_utils
import face_recognition
import pickle
import cv2
from tqdm import tqdm
from PIL import Image

class Extract():

    def __init__(self, opt, output_dir, moods_file, emos_file, ids_file):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.Resize(size=(self._opt.image_size, self._opt.image_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])
        self.output_dir = output_dir
        self.ids = self._read_ids(ids_file)
        self.moods = self._read_moods(moods_file)
        self.emos = self._read_moods(emos_file)





    def _read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return ids

    def _read_moods(self, file_path):
        with open(file_path, 'rb') as f:
            mood_dict = pickle.load(f)
        return mood_dict

    def _read_emos(self, file_path):
        with open(file_path, 'rb') as f:
            emos_dict = pickle.load(f)
        return emos_dict

    def morph_file(self, img_folder, img_name):
        img_path = os.path.join(img_folder, img_name)
        img = cv_utils.read_cv2_img(img_path)
	if img is None:
	    print('%s could not be read' % img_path)
	    return
        expression, emo = self.generate_random_cond()
        morphed_img = self._img_morph(img, expression)
        #output_name = os.path.join(self._opt.output_dir, '{0}_epoch_{1}_intensity_{2}_out.png'.format(os.path.basename(img_path)[:-4], \
        #str(self._opt.load_epoch), str(expression[0])))
        save_path = os.path.join(self.output_dir, img_name)
        self._save_img(morphed_img, save_path)
        return emo

    def _img_morph(self, img, expression):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = face_utils.resize_face(img)

        morphed_face = self._morph_face(face, expression)

        return morphed_face

    def _morph_face(self, face, expression):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        neutral = torch.unsqueeze(torch.from_numpy(np.zeros(len(expression))), 0)
        expression = torch.unsqueeze(torch.from_numpy(expression), 0)

        test_batch1 = {'real_img': face, 'real_cond': neutral, 'desired_cond': expression, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch1)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['fake_imgs_masked']

    def _save_img(self, img, filename):
        filepath = os.path.join(self.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)

    def generate_random_cond(self):
        cond = None
        emo = None
        while cond is None or emo is None:
            rand_sample_id = self.moods.keys()[np.random.randint(0, len(self.moods) - 1)]
            cond = self.moods[rand_sample_id].astype(np.float64)
            emo = self.emos[rand_sample_id]
            cond += np.random.uniform(-0.1, 0.1, cond.shape)
        return cond, emo


if __name__ == "__main__":
    print('before parsing')
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    print('CUDA:', os.environ['CUDA_VISIBLE_DEVICES'])

    opt = TestOptions().parse()
    print('START')
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)
    extractor = Extract(opt, opt.output_dir, opt.moods_pickle_file, opt.emo_test_file, opt.test_ids_file)
    images_folder = os.path.join(opt.data_dir, opt.test_images_folder)
    emo_dict = dict()
    for id in tqdm(extractor.ids):
        emo = extractor.morph_file(images_folder, id)
        emo_dict[id] = emo
    pickle.dump(emo_dict, open(os.path.join(images_folder, 'emos.pkl'), 'w'))
