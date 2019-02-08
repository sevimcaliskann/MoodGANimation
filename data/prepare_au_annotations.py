import numpy as np
import os
from tqdm import tqdm
import argparse
import glob
import re
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-ia', '--input_aus_filesdir', type=str, help='Dir with imgs aus files')
parser.add_argument('-op', '--output_path', type=str, help='Output path')
parser.add_argument('-of', '--output_file', type=str, help='Output file')
args = parser.parse_args()

def get_data(filepaths):
    data = dict()
    for filepath in tqdm(filepaths):
        content = np.loadtxt(filepath, delimiter=', ', skiprows=1)
	row = content[2:19]
	if len(row) == 17:
            data[os.path.basename(filepath[:-4])] = row

    return data

def save_dict(data, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def main():
    filepaths = glob.glob(os.path.join(args.input_aus_filesdir, '*.csv'))
    filepaths.sort()

    # create aus file
    data = get_data(filepaths)

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
    save_dict(data, os.path.join(args.output_path, args.output_file))


if __name__ == '__main__':
    main()
