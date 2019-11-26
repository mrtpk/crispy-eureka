import os
import h5py
import numpy as np
from itertools import cycle
from random import shuffle
import keras
from glob import glob
import random

def load_h5_file(path, variables=None):

    with h5py.File(path, 'r') as hf:
        data = np.array(hf["array"])

    if variables is not None:
        return data[..., variables]
    else:
        return data

def generator_from_h5(basedir, variables=None, batch_size=5, jump_after=300):

    local_proc_rand_gen = np.random.RandomState()  # to use in multiprocessing
    X_filelists = sorted(glob(os.path.join(basedir, 'img', '*.h5')))

    n = 0
    while True:
        if n == 0:
            file = X_filelists[local_proc_rand_gen.choice(len(X_filelists), 1)[0]]
            # print(file)
            X = load_h5_file(file, variables=variables)
            Y = load_h5_file(file.replace('img', 'gt'))
            jump_after = X.shape[0] // batch_size

        if batch_size < X.shape[0]:
            replace=False
        else:
            replace=True

        idx_sample = np.random.choice(X.shape[0], batch_size, replace=replace)
        # print(idx_sample)
        Xtemp = X[idx_sample] if batch_size > 1 else X[idx_sample][None, ...]
        Ytemp = Y[idx_sample] if batch_size > 1 else Y[idx_sample][None, ...]

        n += 1
        if n == jump_after:
            n = 0

        yield Xtemp, Ytemp

class DataLoaderGenerator(keras.utils.Sequence):
    def __init__(self, basedir, batch_size=5, variables=None, shuffle=True, n_samples_per_file=300):
        'Initialization'
        self.batch_size = batch_size
        self.variables = variables
        self.basedir = basedir
        self.list_files = sorted(glob(os.path.join(basedir, 'img', '*.h5')))
        # self.n_classes = n_classes
        self.shuffle = shuffle
        self.n_samples_per_file = n_samples_per_file
        self.local_proc_rand_gen = np.random.RandomState()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        ## it should be like n_sample / batch_size
        # 300 is hard coded
        return int(np.floor(len(self.list_files) * self.n_samples_per_file) / self.batch_size)

    def __getitem__(self, index):

        # generate indexes of the batch
        # indexes = self.indexes[index * self.batch_size: (index+1) * self.batch_size]
        indexes = np.random.choice(self.X.shape[0], self.batch_size)
        # X = np.empty((self.batch_size, *self.dim, len(self.variables)))
        # y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)
        # get data
        X = self.X[indexes]
        y = self.Y[indexes].astype(np.int)

        return X, y

    def on_epoch_end(self):
        'update indexes after each epoch'
        file = self.list_files[self.local_proc_rand_gen.choice(len(self.list_files), 1)[0]]
        # print(file)
        self.X = load_h5_file(file, variables=self.variables)
        self.Y = load_h5_file(file.replace('img', 'gt'))

        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
