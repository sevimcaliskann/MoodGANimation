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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import math

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
        self._moods = get_moods_from_pickle('/home/sevim/Downloads/master_thesis_study_documents/code-examples/affwild/annotations/data.pkl')


    def morph_file(self, img_path, expression, traj):
        img = cv_utils.read_cv2_img(img_path)
        morphed_video = self._img_morph(img, expression)

        output_name = os.path.join(self._opt.output_dir, \
                '{0}_epoch_{1}_out.mp4'.format(os.path.basename(img_path)[:-4], \
                str(self._opt.load_epoch)))
        out = cv2.VideoWriter(output_name,cv2.VideoWriter_fourcc(*'mjpg'), 2, \
                    (morphed_video.shape[2]+morphed_video.shape[1],morphed_video.shape[1]))
        for i in range(morphed_video.shape[0]):
            ch = morphed_video[i]
            f = cv2.resize(traj[i+1], (ch.shape[0], ch.shape[0]))
            tmp = np.concatenate((ch,f), axis=1)
            out.write(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        out.release()
        print('Morphed image is saved at path {}'.format(output_name))

    def morph_and_tile(self, img_path, expression, label):
        img = cv_utils.read_cv2_img(img_path)
        morphed_video = self._img_morph(img, expression, label=label)

        output_name = os.path.join(self._opt.output_dir, \
                '{0}_epoch_{1}_{2}_out.png'.format(os.path.basename(img_path)[:-4], \
                str(self._opt.load_epoch), label))
        tiled_img = np.concatenate(morphed_video, axis=1)
        self._save_img(tiled_img, output_name)
        print('Morphed image is saved at path {}'.format(output_name))


    def _img_morph(self, img, expression, label='concat'):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = face_utils.resize_face(img)

        morphed_face = self._morph_face(face, expression, label)

        return morphed_face

    def _morph_face(self, face, expression, label='concat'):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expression = torch.unsqueeze(torch.from_numpy(expression), 0)
        neutral = torch.unsqueeze(torch.from_numpy(np.array([0.5, 0.5])), 0)
        test_batch1 = {'first_frame': face, 'annotations': expression, 'first_ann': neutral, 'frames': face}
        self._model.set_input(test_batch1)
        imgs1 = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs1[label]

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

def get_moods_from_pickle(path):
    return pickle.load(open(path, 'rb'))

def animate_traj(exp):
    fig = plt.figure(figsize=(10,10))
    canvas = FigureCanvas(fig)
    axes = plt.gca()
    axes.set_xlim([-1.0,1.0])
    axes.set_ylim([-1.0,1.0])
    plt.xlabel('Valence')
    plt.ylabel('Arousal')

    frames = list()
    val = list()
    aro = list()
    for line in exp:
        val.append(line[0])
        aro.append(line[1])
        plt.scatter(val, aro, s=500, c='blue')
        img = fig2data(fig, canvas)
        frames.append(img)

    return frames



def fig2data ( fig, canvas ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = canvas.get_width_height()
    buf = np.fromstring(canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( w, h,3 )

    return buf

def main():
    print('BEGINING')
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)
    print("morph objetc is created")
    image_path = opt.input_path
    expression_ind = np.random.randint(0, len(morph._moods) - 1)
    expression = morph._moods[expression_ind:min(expression_ind+opt.frames_cnt, len(morph._moods))]

    #val = np.expand_dims(np.arange(0,1,0.1), axis=1)
    #val = np.expand_dims(np.arange(0,1,0.1), axis=1)
    #aro = np.expand_dims(-1*np.arange(0,1,0.1), axis=1)

    #third = np.expand_dims(np.zeros(opt.frames_cnt), axis=1)
    #fourth = np.expand_dims(-1*np.ones(opt.frames_cnt), axis=1)
    #expression = np.concatenate((expression, third, fourth), axis=1)
    #expression = np.concatenate((val,aro, third, fourth), axis=1)
    #expression = np.concatenate((val,aro), axis=1)

    frames = animate_traj(expression)
    traj_img_path = os.path.join(opt.output_dir,'traj_out.png')
    morph._save_img(frames[-1], traj_img_path)

    morph.morph_file(image_path, expression, frames)
    morph.morph_and_tile(image_path, expression, 'fake_imgs_masked')
    morph.morph_and_tile(image_path, expression, 'fake_imgs')
    morph.morph_and_tile(image_path, expression, 'img_mask')
    #while expression_ind < len(morph._moods):
        #expression = np.zeros(2)
        #expression[0] = i
        #expression[1] = i
        #morph.morph_file(image_path, expression)
        #expression_ind +=1
        #i +=1
    #expression = np.random.uniform(0, 1, opt.cond_nc)
    #expression = generate_random_cond(conds)




if __name__ == '__main__':
    main()
