import torch
import torchvision.transforms as transforms
import os
import argparse
import glob
import cv2
from utils import face_utils
from utils import cv_utils
import face_recognition
from PIL import Image
import pickle
import numpy as np
from models.models import ModelsFactory
from data.dataset import DatasetFactory
from data.dataset_aus import AusDataset
from data.custom_dataset_data_loader import CustomDatasetDataLoader
from options.feature_options import FeatureOptions
from skimage import io
from tqdm import tqdm

class MiddleLayerFeatures:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self.ids = self.read_ids(os.path.join(self._opt.data_dir, self._opt.input_file))
        self.conds = self.read_conds(os.path.join(self._opt.data_dir, self._opt.aus_file))
        self.ids = list(set(self.ids).intersection(set(self.conds.keys())))
        self.layers = self._opt.layers
        print('#images: ', len(self.ids))
        if not os.path.exists(self._opt.output_dir):
            os.makedirs(self._opt.output_dir)

        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def read_ids(self, file_path):
        ids = np.loadtxt(file_path, delimiter='\t', dtype=np.str)
        return [id[:-4] for id in ids]

    def read_conds(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def get_features_all(self):
        #self.features = dict()
        for id in tqdm(self.ids):
            output_path = os.path.join(self._opt.output_dir, id+'.pkl')
            if os.path.exists(output_path):
                continue
            #cond = self.conds[id]
            cond = np.zeros((17,))
            filepath = os.path.join(self._opt.data_dir, self._opt.images_folder)
            filepath = os.path.join(filepath, id+'.jpg')
            features = self.get_features(filepath, cond)
            if features is None:
                continue
            #new_id = id
            self.save_features(features, output_path)
            print('Saved features at: ', output_path)
            #self.features[new_id] = features

    def save_features(self, features, filepath):
        with open(filepath, 'w') as file:
            pickle.dump(features, file)

    def get_features(self, img_path, expression):
        img = cv_utils.read_cv2_img(img_path)
        if img is None:
            print('Failing to read sample: ', img_path)
            return None
        features = self.img_forward(img, expression)
        return features

    def get_name(self, book, layer):
        layer_name = type(layer).__name__
        count = 0
        key = layer_name + ':' + str(count)
        while key in book.keys():
            count = count+1
            key = layer_name + ':' + str(count)
        return key


    def img_forward(self, img, expression):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = face_utils.resize_face(img)

        morphed_face = self.face_forward(face, expression)

        return morphed_face

    def face_forward(self, face, expression):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expression = expression.astype(dtype = np.float32)
        expression = torch.unsqueeze(torch.from_numpy(expression), 0)

        expression = expression.unsqueeze(2).unsqueeze(3)
        expression = expression.expand(expression.size(0), expression.size(1), face.size(2), face.size(3))
        face = torch.cat([face, expression], dim=1)
        face = face.cuda(self._model._gpu_ids[0], async=True)

        features = dict()
        dump_dict = dict()
        for layer in self._model._G.main:
            key = self.get_name(dump_dict, layer)
            f = layer(face)
            dump_dict[key] = f
            #if key=='ResidualBlock:2' or key=='ResidualBlock:3':
            if key in self.layers:
                features[key] = f.cpu().data.numpy()
            face = f

        img_reg = self._model._G.img_reg(face)
        img_reg = np.squeeze(np.array(img_reg.cpu().detach().numpy()*255, dtype=np.uint8))
        att_reg = self._model._G.attetion_reg(face)
        att_reg = np.squeeze(np.array(att_reg.cpu().detach().numpy()*255, dtype=np.uint8))
        if 'attention' in self.layers:
            features['attention'] = att_reg
        if 'img_reg' in self.layers:
            features['img_reg'] = img_reg

        return features

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # close for cropped_28_01
        cv2.imwrite(filepath, img)


def generate_random_cond(conds):
    cond = None
    while cond is None:
        rand_sample_id = conds.keys()[np.random.randint(0, len(conds) - 1)]
        cond = conds[rand_sample_id].astype(np.float64)
        cond += np.random.uniform(-0.1, 0.1, cond.shape)
    return cond



def main():
    opt = FeatureOptions().parse()
    obj = MiddleLayerFeatures(opt)
    obj.get_features_all()
    #obj.save_features()




if __name__ == '__main__':
    main()
