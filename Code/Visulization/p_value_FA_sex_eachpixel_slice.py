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

def FA(DTI):
    eigenvalue, _ = np.linalg.eig(DTI)
    # pdb.set_trace()
    eigenvalue = eigenvalue.real

    fracdown = np.sqrt((eigenvalue**2).sum(axis = -1))+1e-10

    meaneigenvalue = eigenvalue.mean(axis = -1, keepdims = True)

    fracup = np.sqrt(((eigenvalue-meaneigenvalue)**2).sum(axis = -1))

    FA_value = (np.sqrt(3./2)*fracup/fracdown)
    
    # pdb.set_trace()
    return FA_value.real

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
    DTI = data['dti'].squeeze()

    FA_ = FA(DTI)
    np.save('../data/FA.npy', FA_.mean(0))
    
    PosFA = []
    NegFA = []

    for i in range(len(names)):
        if Dict_[names[i]]:
            PosFA.append(FA_[i])
        else:
            NegFA.append(FA_[i])
    PosFA = np.stack(PosFA)
    NegFA = np.stack(NegFA)
    return [PosFA, NegFA]

def permutation(ODF_):
    mean_pos = ODF_[0].mean(axis = 0)
    mean_neg = ODF_[1].mean(axis = 0)

    pos_num = len(ODF_[0])
    neg_num = len(ODF_[1])

    ori_distance = ((mean_pos-mean_neg)**2)

    print('Ori:', ori_distance)

    whole_data = np.concatenate(ODF_, axis = 0)
    mean_pos = whole_data[0:pos_num].mean(axis = 0)
    mean_neg = whole_data[pos_num:].mean(axis = 0)
    ori_distance = ((mean_pos - mean_neg)**2)
    print('Ori:', ori_distance)

    perm_dist = []
    for i in range(10000):
        np.random.shuffle(whole_data)
        mean_pos_perm = whole_data[0:pos_num].mean(axis = 0)
        mean_neg_perm = whole_data[pos_num:].mean(axis = 0)
        dist = ((mean_pos_perm - mean_neg_perm)**2)
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

    FA_ = divide_group(data, SexDict)

    p_value, perm_dist, ori_distance = permutation(FA_)


    np.savez('../data/FA_Results.npz', p_ori = p_value, perm_dist_ori = perm_dist, dist_ori =ori_distance)

    toc = time.time()
    print(toc-tic)
