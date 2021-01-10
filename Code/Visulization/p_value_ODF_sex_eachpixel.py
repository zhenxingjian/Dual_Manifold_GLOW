import numpy as np
import nibabel as nib
import dipy
import os
import time
import pandas as pd
import multiprocessing as mp
import pdb
import csv
import json

def Load_group(files = []):
    AgeDict = {}
    SexDict = {}
    for file in files:
        count = 0
        with open (file) as csvfile:
            lines = csv.reader(csvfile)
            for row in lines:
                if count == 0:
                    count += 1
                else:
                    AgeDict[row[0]] = '31' in row[4] or '36' in row[4] # elder is true
                    SexDict[row[0]] = 'M' in row[3]
                    count += 1
    return AgeDict, SexDict

def divide_group(data, Dict_):
    with open('../Preprocessing/test_files.json', "r") as f:
        names = json.load(f)
    ODFs = data['odf'].squeeze()

    RODFs = data['rodf'].squeeze()
    PosODF = []
    NegODF = []

    PosRODF = []
    NegRODF = []

    for i in range(len(names)):
        if Dict_[names[i]]:
            PosODF.append(ODFs[i])
            PosRODF.append(RODFs[i])
        else:
            NegODF.append(ODFs[i])
            NegRODF.append(RODFs[i])
    PosODF = np.stack(PosODF)
    NegODF = np.stack(NegODF)
    PosRODF = np.stack(PosRODF)
    NegRODF = np.stack(NegRODF)
    return [PosODF, NegODF], [PosRODF, NegRODF]

def permutation(ODF_):
    mean_pos = ODF_[0].mean(axis = 0)
    mean_neg = ODF_[1].mean(axis = 0)

    pos_num = len(ODF_[0])
    neg_num = len(ODF_[1])

    ori_distance = ((mean_pos-mean_neg)**2).sum(axis = -1)

    print('Ori:', ori_distance)

    whole_data = np.concatenate(ODF_, axis = 0)
    mean_pos = whole_data[0:pos_num].mean(axis = 0)
    mean_neg = whole_data[pos_num:].mean(axis = 0)
    ori_distance = ((mean_pos - mean_neg)**2).sum(axis = -1)
    print('Ori:', ori_distance)

    perm_dist = []
    for i in range(10000):
        np.random.shuffle(whole_data)
        mean_pos_perm = whole_data[0:pos_num].mean(axis = 0)
        mean_neg_perm = whole_data[pos_num:].mean(axis = 0)
        dist = ((mean_pos_perm - mean_neg_perm)**2).sum(axis = -1)
        print(i)
        perm_dist.append(dist)
    perm_dist = np.asarray(perm_dist)

    p_value = 1-(perm_dist<ori_distance).mean(axis = 0)

    return p_value, perm_dist, ori_distance

if __name__ == '__main__':
    files = ['../Preprocessing/HCP_info.csv']
    AgeDict, SexDict = Load_group(files)

    data = np.load('../data/results.npz')

    tic = time.time()

    ODF_, RODF_ = divide_group(data, SexDict)

    p_value, perm_dist, ori_distance = permutation(ODF_)
    p_value_gen, perm_dist_gen, ori_distance_gen = permutation(RODF_)


    np.savez('../data/ODF_Results.npz', p_ori = p_value, p_gen = p_value_gen)

    toc = time.time()
    print(toc-tic)
    pdb.set_trace()
