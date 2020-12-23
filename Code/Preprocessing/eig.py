import numpy as np
import os

PATH = '../HCP_processed/'
files = os.listdir(PATH)

for file in files:
    pth = PATH + file
    data = np.load(pth)
    eig,_ = np.linalg.eig(data['DTI'])
    dti = data['DTI']
    odf = data['ODF']
    odf = odf/(1e-5 + odf.sum(-1, keepdims=True))
    mask = data['mask']
    name = data['name']

    np.savez(pth, DTI=dti, ODF = odf, Eig = eig, mask = mask, name = name)
    print(name)
    