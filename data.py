# This directory contains code for downloading and proceesing data

import pickle
import os
import time
import glob
import numpy as np

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == "__main__":
    print(f"{bcolors.OKBLUE}Starting data.py{bcolors.ENDC}\n")

    if not os.path.isdir('data'):
        os.system('mkdir data')

    if os.path.isfile('cifar-10-python.tar.gz'):
        print("cifar-10-python.tar.gz already in directory. Skipping Download ...")
    else:
        os.system("wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")

    os.system("tar -xvzf cifar-10-python.tar.gz")

    data_files = glob.glob('cifar-10-batches-py/data*')
    test_file  = glob.glob('cifar-10-batches-py/test*')

    labels = []
    data_arrays = []

    for path in data_files:
        dt = unpickle(path)
        data, label = dt[b'data'], dt[b'labels']
        data = data.reshape([10000,32,32,3])
        data_arrays.append(data)
        labels += label

    data = np.concatenate(data_arrays)
    labels = np.array(labels)
    
    np.save('data/images.npy', data)
    np.save('data/labels.npy', labels)

    print(f"\n{bcolors.OKGREEN}Data Saved to ./data/ directory in npy format{bcolors.ENDC}")