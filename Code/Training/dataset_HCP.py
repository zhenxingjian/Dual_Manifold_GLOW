import pickle
import torch
import torch.nn as nn
import numpy as np
import random
import sys
import time
import os
import math
import json
import datetime
import matplotlib.pyplot as plt
import math
import copy
from collections import OrderedDict

import json
import sys
from multiprocessing import Pool

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class Dataset_GLOW(torch.utils.data.Dataset):
    '''
    The dataloader of the HCP dataset
    Use the json file to control which one to load
    Use numpy to load the data, can be changed into h5 for further improvement
    Current version works well with 32x32x32 size data
    Can reduce the image size into 28x28x28 (tested in our inner experiment)
    '''
    def __init__(self, data_path = '../HCP_processed/' , train = True):
        super(Dataset_GLOW, self).__init__()

        self.data_path = data_path
        self.train = train

        if self.train:
            self.json_path = '../Preprocessing/train_files.json'
        else:
            self.json_path = '../Preprocessing/test_files.json'

        with open(self.json_path, "r") as f:
            files = json.load(f)
        self.files = [os.path.join(self.data_path, file)+'.npz' for file in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        file = self.files[item]
        data_npz = np.load(file)
        DTI = torch.from_numpy(data_npz['DTI']).float()
        ODF = torch.from_numpy(data_npz['ODF']).float()
        EIG = torch.from_numpy(data_npz['Eig']).float()
        mask = torch.from_numpy(data_npz['mask']).float()

        #### normalization ####
        ODF = ODF/(torch.sum(ODF, -1, keepdim=True) + 1e-5)
        #### square root representitive ####
        ODF = torch.sqrt(ODF)

        return DTI, EIG, ODF, mask

class DataLoaderX(DataLoader):
    '''
    Background loading, fully use the I/O and CPU
    '''

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

if __name__ == '__main__':
    dataset = Dataset_GLOW(data_path = '../HCP_processed/', train = True)
    dataloader = DataLoaderX(dataset, batch_size=16, shuffle=True, num_workers=4)

    for data in dataloader:
        dti, eig, odf, msk = data
        print(dti.shape)
        print(eig.shape)
        print(odf.shape)
        print(msk.shape)
        print('----------------------')
