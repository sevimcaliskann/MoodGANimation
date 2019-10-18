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
import imageio

class MorphFacesInTheWild:
    def __init__(self, opt, is_comparison=False):
        self._opt = opt
        self._out_dir = os.path.join(self._opt.output_dir, self._opt.name) if not is_comparison else \
                        os.path.join(self._opt.output_dir, self._opt.comparison_model_name)
        if not os.path.isdir(self._out_dir):
            os.makedirs(self._out_dir)
        if is_comparison:
            self._opt.name = self._opt.comparison_model_name
            self._opt.load_epoch = self._opt.comparison_load_epoch
            self._opt.recurrent = (not is_comparison) & self._opt.recurrent

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.Resize(size=(self._opt.image_size, self._opt.image_size)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])
        self._moods = get_moods_from_pickle(self._opt.moods_pickle_file, self._opt.cond_nc)

    def _img_morph(self, img, expression, label='concat'):
        face = self.crop_face(img)
        morphed_face = self._morph_face(face, expression, label)
        return morphed_face


    def morph_file(self, img_path, expression, img=None):
        if img is None:
            img = cv_utils.read_cv2_img(img_path)
        morphed_video = self._img_morph(img, expression)

        output_name = os.path.join(self._out_dir, \
                '{0}_epoch_{1}_out.gif'.format(os.path.basename(img_path)[:-4], \
                str(self._opt.load_epoch)))
        imageio.mimsave(output_name, list(morphed_video))
        print('Morphed image is saved at path {}'.format(output_name))

    def morph_and_tile(self, img_path, expression, label, img = None):
        if img is None:
            img = cv_utils.read_cv2_img(img_path)

        morphed_video = self._img_morph(img, expression, label=label)

        output_name = os.path.join(self._out_dir, \
                '{0}_epoch_{1}_{2}_out.png'.format(os.path.basename(img_path)[:-4], \
                str(self._opt.load_epoch), label))
        tiled_img = np.concatenate(morphed_video, axis=1)
        self._save_img(tiled_img, output_name)
        print('Morphed image is saved at path {}'.format(output_name))

    def crop_face(self, img):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = face_utils.resize_face(img)
        return face


    def _morph_face(self, face, expression, label='concat'):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expression = torch.unsqueeze(torch.from_numpy(expression), 0)
        neutral = torch.unsqueeze(torch.from_numpy(np.array([0.5, 0.5])), 0)
        test_batch1 = {'first_frame': face, 'annotations': expression, 'first_ann': neutral, 'frames': face}
        self._model.set_input(test_batch1)
        imgs1 = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs1[label]

    def random_generation(self, get_start_from_video=False):
        val = np.expand_dims(np.arange(-1.0,1.0,float(2)/self._opt.frames_cnt), axis=1)
        aro = np.expand_dims(np.arange(-1.0,1.0,float(2)/self._opt.frames_cnt), axis=1)
        #val = np.expand_dims(np.zeros(10), axis=1)

        third = np.expand_dims(np.zeros(self._opt.frames_cnt), axis=1)
        fourth = np.expand_dims(-1*np.ones(self._opt.frames_cnt), axis=1)
        #expression = np.concatenate((val,third, third, fourth), axis=1)
        expression = np.concatenate((aro,val), axis=1) if self._opt.cond_nc == 2 else \
                        np.concatenate((val,third, third, fourth), axis=1)

        traj = animate_traj(expression, self._out_dir)
        if not get_start_from_video:
            self.morph_file(self._opt.input_path, expression)
            self.morph_and_tile(self._opt.input_path, expression, 'fake_imgs_masked')
            self.morph_and_tile(self._opt.input_path, expression, 'fake_imgs')
            self.morph_and_tile(self._opt.input_path, expression, 'img_mask')
        else:
            img, _ = self.get_n_faces_from_video(1)
            self.morph_file(self._opt.groundtruth_video, expression, img=img)
            self.morph_and_tile(self._opt.groundtruth_video, expression, 'fake_imgs_masked', img=img)
            self.morph_and_tile(self._opt.groundtruth_video, expression, 'fake_imgs', img=img)
            self.morph_and_tile(self._opt.groundtruth_video, expression, 'img_mask', img=img)

    def get_n_faces_from_video(self, count):
        video = cv2.VideoCapture(self._opt.groundtruth_video)
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = self._opt.groundtruth_video.split('/')[-1][:-4]
        #start = np.random.randint(0, length-self._opt.frames_cnt)
        start = 1163
        video.set(1, start)

        success, start_face = video.read()
        start_face = self.crop_face(cv2.cvtColor(start_face, cv2.COLOR_RGB2BGR))
        if not success:
            print('video %s cannot be read!' % self._opt.groundtruth_video)
            video.release()
            return None

        ground_faces = list()
        anns = list()
        anns.append(np.expand_dims(np.asarray(self._moods[video_name + str(start+1)]), axis=0))
        ground_faces.append(start_face)
        for i in range(1, count):
            success, face = video.read()
            face = self.crop_face(face)
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            ground_faces.append(face)
            anns.append(np.expand_dims(np.asarray(self._moods[video_name + str(start+i+1)]), axis=0))
        video.release()
        return np.squeeze(np.stack(ground_faces, axis=0)), anns

    def generate_from_groundtruth(self):
        ground_faces, anns = self.get_n_faces_from_video(self._opt.frames_cnt)
        tiled_ = np.concatenate(ground_faces, axis=1) if self._opt.frames_cnt>1 else ground_faces
        tiled_name = os.path.join(self._out_dir, \
                '{0}_epoch_{1}_{2}_out.png'.format(os.path.basename(self._opt.groundtruth_video)[:-4], \
                str(self._opt.load_epoch), 'groundtruth'))
        self._save_img(tiled_, tiled_name)
        imageio.mimsave(tiled_name[:-4]+'.gif', ground_faces)
        start_face = ground_faces[0] if self._opt.frames_cnt>1 else np.copy(ground_faces)

        anns = np.concatenate(anns, axis=0)
        traj = animate_traj(anns, self._out_dir)

        self.morph_file(self._opt.groundtruth_video, anns, img=start_face)
        self.morph_and_tile(self._opt.groundtruth_video, anns, 'fake_imgs_masked', img = start_face)
        self.morph_and_tile(self._opt.groundtruth_video, anns, 'fake_imgs', img=start_face)
        self.morph_and_tile(self._opt.groundtruth_video, anns, 'img_mask', img=start_face)
        return start_face, anns

    def _save_img(self, img, filename):
        filepath = os.path.join(self._out_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # close for cropped_28_01
        cv2.imwrite(filepath, img)

def get_moods_from_pickle(path, cond_nc):
    data = pickle.load(open(path, 'rb'))
    if cond_nc==2:
        moods = dict()
        for key, val in data.items():
            key = key.split('/')[-1][:-4]
            moods[key] = val
        return moods
    else:
        return data

def animate_traj(exp, save_path):
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

    traj_img_path = os.path.join(save_path,'traj_out.png')
    img = cv2.cvtColor(frames[-1], cv2.COLOR_RGB2BGR) # close for cropped_28_01
    cv2.imwrite(traj_img_path, img)
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
    morph = MorphFacesInTheWild(opt)
    morph.random_generation(False)
    img, expression = morph.generate_from_groundtruth()

    #morph = MorphFacesInTheWild(opt, is_comparison=True)
    #morph.morph_file(opt.groundtruth_video, expression, img=img)
    #morph.random_generation(False)


    #opt.name = 'maximize_mask_no_cycle'
    #opt.load_epoch = 50
    #opt.recurrent=False
    #opt.moods_pickle_file = '/home/sevim/Downloads/master_thesis_study_documents/code-examples/affwild/annotations/384_4d.pkl'
    #opt.cond_nc = 4
    #morph = MorphFacesInTheWild(opt, is_comparison=False)
    #morph.morph_file(opt.groundtruth_video, expression, img=img)
    #morph.random_generation(False)
    #morph.morph_and_tile(opt.groundtruth_video, expression, 'fake_imgs_masked', img=img)
    #morph.morph_and_tile(opt.groundtruth_video, expression, 'fake_imgs', img=img)
    #morph.morph_and_tile(opt.groundtruth_video, expression, 'img_mask', img=img)
    #morph.random_generation(False)



if __name__ == '__main__':
    main()
