from __future__ import print_function
from PIL import Image
from tqdm import tqdm
from numba import jit, autojit
import os
import glob
import numpy as np
import openpyxl
import pickle
import urllib, cStringIO
import klepto
import pandas as pd
#%load_ext cythonmagic


'''def write_https(mypath ='/srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws'):
    current_dir = os.getcwd()
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f.endswith('.txt')]
    for f in tqdm(onlyfiles):
        with open(os.path.join(mypath, f)) as content_file:
            content = content_file.read()
	    lines = content.split('\n')[0:-1]
	    https = [[line.split('\t')[0]] for line in lines]
	    file_name = f[0:-4] + '_https.txt'
	    with open(os.path.join(mypath, file_name), 'w') as out:
            for line in tqdm(https):
                out.write(line[0] + '\n')
        print('%s is finished!' % f)
    os.chdir(current_dir)'''


def download_dataset_images(mypath ='/srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws'):
    current_dir = os.getcwd()
    onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f.endswith('.txt')]
    for f in tqdm(onlyfiles):
    	dir_name = f[:-4]
    	image_dir = os.path.join(mypath, dir_name)
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
    	os.chdir(image_dir)
        with open(os.path.join(mypath, f)) as content_file:
            for line in tqdm(content_file):
                download_image(line)
        print('%s is finished!' % f)
    os.chdir(current_dir)


def download_image(line):
    if(line!=''):
        url = line.split('\t')[0]
        image_name = url.split('/')[-1]
        urllib.urlretrieve(url, image_name)




def read_aus_from_xlsx(filepath ='/srv/glusterfs/csevim/datasets/emotionet/EmotioNet_FACS_aws_without_passw.xlsx', make_unknown_zero=False):
    d = klepto.archives.dir_archive(os.path.join(os.path.dirname(filepath), 'dataset'), cached=True, serialized=True)
    au_ids = np.array([1,2,4,5,6,9,12,17,20,25,26])
    #au_ids = np.array([1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45])
    au_ids += 1
    xl_file = pd.ExcelFile(filepath)
    for sheet_name in xl_file.sheet_names:
      chunk = xl_file.parse(sheet_name)
      urls = chunk[chunk.columns[0]]
      names = np.array([url.split('/')[-1][:-5] for url in urls], dtype=np.unicode)
      aus = np.array(chunk[np.take(chunk.columns, au_ids)], dtype=np.float32)
      if make_unknown_zero:
          aus[aus==999] = 0
      dictionary = dict(zip(names, aus))
      d['data'] = dictionary
      d.dump()
      d.clear()

def create_dataset(mypath='/srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws', make_unknown_zero=False):
    if not os.path.exists(os.path.join(mypath, 'dataset')):
        os.makedirs(os.path.join(mypath, 'dataset'))
    d = klepto.archives.dir_archive(os.path.join(mypath, 'dataset'), cached=True, serialized=True)
    txts_path = os.path.join(mypath, "txts")
    files = [f for f in os.listdir(txts_path) if os.path.isfile(os.path.join(txts_path, f)) and f.endswith('.txt')]
    for f in tqdm(files):
        append_to_dataset(d, os.path.join(txts_path, f), make_unknown_zero)

def append_to_dataset(archive, mypath = '/srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/dataFile_1001.txt', make_unknown_zero= False ):
    with open(mypath, 'r') as content_file:
        content = content_file.read()
    rows = content.split('\n')[:-1]
    names = np.array([row.split('\t')[0].split('/')[-1][:-4] for row in rows], dtype=np.unicode)
    au_ids = np.array([1,2,4,5,6,9,12,17,20,25,26])
    #au_ids = np.array([1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45])
    au_ids += 1
    aus = np.array([np.take(row.split('\t'), au_ids) for row in rows], dtype=np.int32)
    if make_unknown_zero:
        aus[aus==999] = 0
    dictionary = dict(zip(names, aus))
    dirname = mypath.split('/')[-1][:-4]
    archive[dirname] = dictionary
    archive.dump()
    archive.clear()

def read_affectnet_labels(filepath ='/srv/glusterfs/csevim/datasets/affectnet/training.csv'):
    df = pd.read_csv(filepath)
    dataset = df[['expression', 'valence', 'arousal']]
    ids = df[['subDirectory_filePath']]
    dataset = np.asarray(dataset.values.tolist(), dtype = np.unicode)
    ids = np.unique(np.asarray(ids.values.tolist(), dtype = np.unicode))
    dictionary = dict(zip(ids, dataset))
    file = open(os.path.join(os.path.dirname(filepath), 'affectnet_dict.pkl'), 'wb')
    pickle.dump(dictionary, file)
    file.close()
    return dictionary

def save_train_test_ids(train_ratio = 0.8, filepath ='/srv/glusterfs/csevim/datasets/emotionet/EmotioNet_FACS_aws_without_passw.xlsx'):
    names = []
    if filepath[-4:]=='xlsx':
        count = 0
        xl_file = pd.ExcelFile(filepath)
        for sheet_name in xl_file.sheet_names:
          chunk = xl_file.parse(sheet_name)
          urls = chunk[chunk.columns[0]]
          names_sheet = np.array([url.split('/')[-1][:-1] for url in urls], dtype=np.unicode)
          if count ==0:
              names = names_sheet
          else:
              names = np.append(names, names_sheet, axis=0)
              count = count+1
    elif filepath[-3:]=='txt':
        with open(filepath, 'r') as content_file:
            content = content_file.read()
        rows = content.split('\n')[:-1]
        names = np.array([row.split('\t')[0].split('/')[-1] for row in rows], dtype=np.unicode)
    indices = np.arange(len(names))
    np.random.shuffle(indices)
    indices = indices[:250000]
    train_num = int(len(indices)*train_ratio)
    train_ids = indices[:train_num]
    test_ids = indices[train_num:]
    train_samples = names[train_ids]
    test_samples = names[test_ids]

    save_to_csv('train_ids_frac.csv', train_samples)
    save_to_csv('test_ids_frac.csv', test_samples)

def save_to_csv(path, data):
    if not os.path.exists(path):
        writeFile = open(path, 'w')
    else:
        writeFile = open(path, 'a')
    [writeFile.write(row+'\n') for row in data]
    writeFile.close()



def save_test_train_multi(mypath='/srv/glusterfs/csevim/datasets/emotionet/emotioNet_challenge_files_server_challenge_1.2_aws/txts', train_ratio=0.8):
    files = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f)) and f.endswith('.txt')]
    names = np.array([])
    for f in tqdm(files):
	filepath = os.path.join(mypath, f)
        with open(filepath, 'r') as content_file:
            content = content_file.read()
        rows = content.split('\n')[:-1]
        if names.size==0:
            names = np.array([row.split('\t')[0].split('/')[-1] for row in rows], dtype=np.unicode)
        else:
            names = np.concatenate((names, np.array([row.split('\t')[0].split('/')[-1] for row in rows], dtype=np.unicode)), axis=0)
    indices = np.arange(len(names))
    np.random.shuffle(indices)
    indices = indices[:250000]
    train_num = int(len(indices)*train_ratio)
    train_ids = indices[:train_num]
    test_ids = indices[train_num:]
    train_samples = names[train_ids]
    test_samples = names[test_ids]
    save_to_csv('train_ids_frac.csv', train_samples)
    save_to_csv('test_ids_frac.csv', test_samples)
