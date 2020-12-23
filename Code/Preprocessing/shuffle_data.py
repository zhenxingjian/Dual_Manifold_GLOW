import os
import numpy as np
import pdb

IDs = np.loadtxt(fname = "./All_HCP_ID.txt", dtype=str)

Batch_num = int(len(IDs))

if not os.path.exists('../HCP_processed/'):
    os.mkdir('../HCP_processed/')

for i in range(Batch_num):
    start = i 
    Ids_In_Batch = IDs[start]
    DTIs = []
    ODFs = []
    masks = []
    names = []
    
    if os.path.exists('../HCP/'+Ids_In_Batch):
        pth = '../HCP/'+Ids_In_Batch + '/T1w/Diffusion/'
    npzfile = pth + 'processed_data.npz'
    npz_ = np.load(npzfile, allow_pickle=True)

    ###### [2:34, 6:38, 2:34] will crop the center of   ######
    ###### the 1/4 resolution brain image               ######
    # hard-coded here, but may be changed accordingly
    DTIs=npz_['DTI'][2:34, 6:38, 2:34, ...]
    ODFs=npz_['ODF'][2:34, 6:38, 2:34, ...]
    masks=npz_['mask'][2:34, 6:38, 2:34, ...]
    names=npz_['name']
    print(Ids_In_Batch)
    np.savez('../HCP_processed/' + str(npz_['name']), DTI = DTIs, ODF = ODFs, mask = masks, name = names)