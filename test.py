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
        expression = torch.unsqueeze(torch.from_numpy(expression/5.0), 0)



        real_cond = np.zeros(17)
        real_cond[7] = 0.5
        real_cond = torch.unsqueeze(torch.from_numpy(real_cond), 0)
        test_batch1 = {'real_img': face, 'real_cond': expression, 'desired_cond': real_cond, 'sample_id': torch.FloatTensor(), 'real_img_path': []}


        real_cond = np.zeros(17)
        real_cond[8] = 0.5
        real_cond = torch.unsqueeze(torch.from_numpy(real_cond), 0)
        test_batch2 = {'real_img': face, 'real_cond': expression, 'desired_cond': real_cond, 'sample_id': torch.FloatTensor(), 'real_img_path': []}

        test_batch3 = {'real_img': face, 'real_cond': expression, 'desired_cond': expression, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        #test_batch = {'real_img': face, 'real_cond': expression, 'desired_cond': expression, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch1)
        imgs1, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        self._model.set_input(test_batch2)
        imgs2, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        self._model.set_input(test_batch3)
        imgs3, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)



        mask1 = imgs1['fake_img_mask']
        mask2 = imgs2['fake_img_mask']
        mask3 = imgs3['fake_img_mask']
        fake1 = imgs1['fake_imgs']
        fake2 = imgs2['fake_imgs']
        fake3 = imgs3['fake_imgs']
        real = imgs1['real_img']



        output_name = os.path.join(self._opt.output_dir, 'mask1.png')
        self._save_img(mask1, output_name)
        output_name = os.path.join(self._opt.output_dir, 'mask2.png')
        self._save_img(mask2, output_name)
        output_name = os.path.join(self._opt.output_dir, 'mask3.png')
        self._save_img(mask3, output_name)
        output_name = os.path.join(self._opt.output_dir, 'fake1.png')
        self._save_img(fake1, output_name)
        output_name = os.path.join(self._opt.output_dir, 'fake2.png')
        self._save_img(fake2, output_name)
        output_name = os.path.join(self._opt.output_dir, 'fake3.png')
        self._save_img(fake3, output_name)
        output_name = os.path.join(self._opt.output_dir, 'real.png')
        self._save_img(imgs1['real_img'] , output_name)

        '''mask1 = cv2.normalize(mask1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mask2 = cv2.normalize(mask2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        test1 = np.array(mask1*real + (1-mask1)*fake1)
        test2 = np.array(mask2*real + (1-mask2)*fake2)
        test3 = np.array(mask1*real + (1-mask1)*fake2)
        test4 = np.array(mask2*real + (1-mask2)*fake1)
        print('test1 shape: ', test1.shape)
        print('test2 shape: ', test2.shape)
        print('test3 shape: ', test3.shape)
        print('test4 shape: ', test4.shape)

        self._save_img(test1 , os.path.join(self._opt.output_dir, 'test1.png'))
        self._save_img(test2 , os.path.join(self._opt.output_dir, 'test2.png'))
        self._save_img(test3 , os.path.join(self._opt.output_dir, 'test3.png'))
        self._save_img(test4 , os.path.join(self._opt.output_dir, 'test4.png'))'''
        return imgs1['concat']

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

def get_aus_values(image_path, csv_folders):
    img_base = os.path.basename(image_path)
    img_base = img_base[:-4]
    csv_path = os.path.join(csv_folders, img_base + '.csv')
    tmp = np.loadtxt(csv_path, delimiter='\n', dtype = np.str)[1]
    aus = np.array(tmp.split(','), dtype=np.float32)[2:19]
    return aus



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
    '''expression = np.zeros(17)
    for i in [0, 0.2, 0.4, 0.5, 0.8, 1.0]:
        expression[opt.au_index] = i
        print('expression: ', expression)
        morph.morph_file(image_path, expression)'''
    #expression = np.zeros(17)
    expression = get_aus_values(image_path, opt.aus_csv_folder)
    print('expression: ', expression)
    morph.morph_file(image_path, expression)
    #expression = np.random.uniform(0, 1, opt.cond_nc)
    #expression = generate_random_cond(conds)




if __name__ == '__main__':
    main()
