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
import operator
import math

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
        self.aus_dict = dict()
        self.aus_dict['happy'] = [8,14]
        self.aus_dict['sad'] = [2,10]
        self.aus_dict['fearful'] = [0,2,12,14]
        self.aus_dict['angry'] = [2,5] # 24 is not encoded in action units??
        self.aus_dict['surprised'] = [0,1,14,15]
        self.aus_dict['disgusted'] = [6, 7, 11]
        self.aus_dict['happily_surprised'] = [0,1,8,14]
        self.aus_dict['happily_disgusted'] = [7,8,14]
        self.aus_dict['sadly_disgusted'] = [2,7]
        self.aus_dict['fearfully_angry'] = [2,12,14]
        self.aus_dict['fearfully_surprised'] = [0,1,3,12,14]
        self.aus_dict['sadly_angry'] = [2,5,10]
        self.aus_dict['angrily_surprised'] = [2,14,15]
        self.aus_dict['appalled'] = [2,6,7]
        self.aus_dict['angrily_disgusted'] = [2,7,11]
        self.aus_dict['awed'] = [0,1,3,14]
        self.labels = {'angrily_disgusted':0, 'angrily_surprised':1, 'angry':2, \
        'appalled':3, 'awed':4, 'disgusted':5, \
        'fearful':6, 'fearfully_angry':7, 'fearfully_surprised':8, \
        'happily_disgusted':9, 'happily_surprised':10, 'happy':11, \
        'sad':12, 'sadly_angry':13, 'sadly_disgusted':14, 	'surprised':15}
        #self.emos = self._read_moods(emos_file)




    def get_emotion_label_from_aus(self, aus_dict, labels, cond):
        conf = dict()
        for k,v in aus_dict.items():
            conf[k] = self.euclidean_distance(cond[v], len(v))

        name = max(conf.iteritems(), key=operator.itemgetter(1))[0]
        label = labels[name]

        return label

    def euclidean_distance(self, arr, n) :
        summation = 0
        for i in range(0,n) :
            summation = summation + arr[i]*arr[i]

        # compute geometric mean through
        # formula pow(product, 1/n) and
        # return the value to main function.
        gm = (float)(math.pow(summation, 0.5))
        return gm



    def _read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        ids = [id[:-4] for id in ids]
        #ids = ids[:25000]
        return ids

    def _read_moods(self, file_path):
        with open(file_path, 'rb') as f:
            mood_dict = pickle.load(f)
        self.ids = list(set(mood_dict.keys()).intersection(set(self.ids)))
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
	    return None
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
            cond = cond/5
            #emo = self.emos[rand_sample_id]
            emo = self.get_emotion_label_from_aus(self.aus_dict, self.labels, cond)
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
        emo = extractor.morph_file(images_folder, id+'.jpg')
        if emo!=None:
            emo_dict[id+'.jpg'] = emo
    pickle.dump(emo_dict, open(os.path.join(opt.output_dir, 'emos.pkl'), 'w'))
