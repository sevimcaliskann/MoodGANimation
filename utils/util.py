from __future__ import print_function
from PIL import Image
import numpy as np
import os
import torchvision
import math
import cv2
import torchvision.transforms as transforms

def tensor2im(img, imtype=np.uint8, unnormalize=True, idx=0, nrows=None):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        if img.shape[1]>1:
            img = img.view(-1, 3, img.size(2), img.size(3))
        else:
            img = img.view(-1, 1, img.size(2), img.size(3))
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)

    if unnormalize:
        #mean = [0.5, 0.5, 0.5]
        #std = [0.5, 0.5, 0.5]
        std = [2.0, 2.0, 2.0]
        mean = [-1.0, -1.0, -1.0]

        transform = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
        img = transform(img)

    image_numpy = img.cpu().numpy()
    image_numpy_t = np.transpose(image_numpy, (1, 2, 0))
    image_numpy_t = image_numpy_t*254.0
    return image_numpy_t.astype(imtype)

def tensor2vid(vid, vidtype = np.uint8, unnormalize=True, idx=0, nrows = None):
    nrows = nrows if nrows is not None else int(math.sqrt(vid.size(0)))
    if idx>=0:
        vid = vid[idx]
    else:
        l = list()
        for ii in range(vid.size(1)):
            l.append(torchvision.utils.make_grid(vid[:,ii,:, :, :], nrows))
        vid = torch.stack(l)
    if unnormalize:
        std = [2.0, 2.0, 2.0]
        mean = [-1.0, -1.0, -1.0]

        transform = transforms.Compose([transforms.Normalize(mean=mean, std=std)])
        l = list()
        for b in vid:
            l.append(transform(b))
        vid = torch.stack(l)

    video_numpy = vid.cpu().numpy()
    video_numpy_t = np.transpose(video_numpy, (0, 2, 3, 1))
    video_numpy_t = video_numpy_t*254.0
    return video_numpy_t.astype(imtype)

def tensor2maskim(mask, imtype=np.uint8, idx=0, nrows=1):
    im = tensor2im(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=-1)
    return im

def tensor2maskvid(mask, imtype=np.uint8, idx=0, nrows=1):
    vid = tensor2vid(mask, imtype=imtype, idx=idx, unnormalize=False, nrows=nrows)
    if vid.shape[3] == 1:
        vid = np.repeat(vid, 3, axis=-1)
    return vid

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image_numpy, image_path):
    mkdir(os.path.dirname(image_path))
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_video(video_numpy, video_path):
    mkdir(os.path.dirname(image_path))
    skvideo.io.vwrite(video_path, video_numpy)

def save_str_data(data, path):
    mkdir(os.path.dirname(path))
    np.savetxt(path, data, delimiter=",", fmt="%s")
