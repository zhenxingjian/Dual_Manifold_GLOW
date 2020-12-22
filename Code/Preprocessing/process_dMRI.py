import numpy as np
import nibabel as nib
import dipy
import os
import time
import pandas as pd
import multiprocessing as mp
import pdb

from dipy.segment.mask import median_otsu
from dipy.reconst.shore import ShoreModel
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
import dipy.reconst.dti as dti


def dMRI2ODF_DTI(PATH):
    '''
    Input the dMRI data
    return the ODF
    '''
    print(PATH)
    if os.path.exists(PATH+'processed_data.npz'):
        return None
    dMRI_path = PATH + 'data.nii.gz'
    mask_path = PATH + 'nodif_brain_mask.nii.gz'
    dMRI_img = nib.load(dMRI_path)
    dMRI_data = dMRI_img.get_fdata()
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()

    ########## subsample ##########
    # in the main paper, to train the full 3D brain
    # We downsample the data into around 32x32x32
    # If not downsample, it can process the full-size brain image
    # but cannot fit into GPU memory
    dMRI_data = dMRI_data[::4,::4,::4,...]
    mask = mask[::4,::4,::4,...]

    bval = PATH + "bvals"
    bvec = PATH + "bvecs"

    radial_order = 6
    zeta = 700
    lambdaN = 1e-8
    lambdaL = 1e-8
    
    ###### process the ODF data ######
    # size is 32x32x32x(362)
    gtab = gradient_table(bvals = bval, bvecs = bvec)
    asm = ShoreModel(gtab, radial_order=radial_order,
                     zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    asmfit = asm.fit(dMRI_data, mask = mask)
    sphere = get_sphere('symmetric362')
    dMRI_odf = asmfit.odf(sphere)
    dMRI_odf[dMRI_odf<=0]=0 # remove the numerical issue 

    ###### process the DTI data ######
    # size is 32x32x32x(3x3)
    tenmodel = dti.TensorModel(gtab)
    dMRI_dti = tenmodel.fit(dMRI_data, mask)
    dMRI_dti = dMRI_dti.quadratic_form

    name = PATH.split('/')[2] # here, might be affected by the path
                              # change the [2] here for the correct index
    np.savez(PATH+'processed_data.npz', DTI=dMRI_dti, ODF = dMRI_odf, mask = mask, name = name)

    return None


def main():
    mainPATH = '../HCP/' # The path to the HCP folder
                         # If downloaded correctly, the folder will have the structure:
                         # */HCP/%%%%%%/T1W/Diffusion/
                         # etc.
    PATH = os.listdir(mainPATH)
    ###### remove all the none-folders ######
    del_ = []
    for path in PATH:
        if '.zip' in path or '.np' in path or 'csv' in path:
            del_.append(path)
    for path in del_:
        PATH.remove(path)

    ###### initial processing ######
    name = []
    for i in range(len(PATH)):
        name.append(PATH[i])
        PATH[i] = mainPATH + PATH[i] + '/T1w/Diffusion/'

    for pth in PATH:
        dMRI2ODF_DTI(pth)


if __name__ == '__main__':
    main()

