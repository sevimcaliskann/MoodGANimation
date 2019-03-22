import os
import argparse
import glob
import torch
import torchvision.transforms as transforms
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
from options.test_options import TestOptions
from skimage import io

class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.Resize(size=(self._opt.image_size, self._opt.image_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, expression):
        img = cv_utils.read_cv2_img(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # Open this one for cropped_28_01
        #newX,newY = img.shape[1]*2, img.shape[0]*2
        #img = cv2.resize(img,(newX,newY))
        morphed_img = self._img_morph(img, expression)
        #morphed_img = cv2.cvtColor(morphed_img, cv2.COLOR_RGB2BGR)
        output_name = os.path.join(self._opt.output_dir, '{0}_epoch_{1}_intensity_{2}_out.png'.format(os.path.basename(img_path)[:-4], str(self._opt.load_epoch), str(expression[self._opt.au_index]*10)))
        self._save_img(morphed_img, output_name)
        print('Morphed image is saved at path {}'.format(output_name))


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
        reconst = np.zeros(17)
        reconst = torch.unsqueeze(torch.from_numpy(reconst), 0)
        expression = torch.unsqueeze(torch.from_numpy(expression), 0)
        test_batch1 = {'real_img': face, 'real_cond': reconst, 'desired_cond': expression, 'sample_id': torch.FloatTensor(), 'real_img_path': []}

        self._model.set_input(test_batch1)
        imgs1, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)

        return imgs1['concat']

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # close for cropped_28_01
        cv2.imwrite(filepath, img)



def main():
    print('BEGINING')
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)


    #aus_dataset_obj = AusDataset(opt, False)
    #conds_filepath = opt.test_aus_file
    #conds = aus_dataset_obj._read_conds(conds_filepath)

    morph = MorphFacesInTheWild(opt)
    print("morph objetc is created")
    image_path = opt.input_path
    expression = np.zeros(17)
    for i in [0, 0.2, 0.4, 0.5, 0.8, 1.0]:
        expression[opt.au_index] = i
        print('expression: ', expression)
        morph.morph_file(image_path, expression)




if __name__ == '__main__':
    main()
