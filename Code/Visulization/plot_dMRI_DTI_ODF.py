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

import numpy as np
import nibabel as nib
import dipy
import os
import time
import pandas as pd
import multiprocessing as mp

from dipy.segment.mask import median_otsu
from dipy.reconst.shore import ShoreModel
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
import dipy.reconst.dti as dti
from dipy.viz import window, actor
from dipy.reconst.dti import fractional_anisotropy, color_fa

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def dMRI2ODF_DTI(PATH):
    '''
    Input the dMRI data
    return the ODF
    '''
    dMRI_path = PATH + 'data.nii.gz'
    mask_path = PATH + 'nodif_brain_mask.nii.gz'
    dMRI_img = nib.load(dMRI_path)
    dMRI_data = dMRI_img.get_fdata()
    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata()

    ########## subsample ##########
    # dMRI_data = dMRI_data[45:-48,50:-65,51:-54,...]
    # mask = mask[45:-48,50:-65,51:-54]
    # breakpoint()
    dMRI_data = dMRI_data[:,87,...]
    mask = mask[:,87,...]

    for cnt in range(10):
        fig=plt.imshow(dMRI_data[:,:,cnt].transpose(1,0),cmap='Greys',interpolation='nearest')
        plt.axis('off')
        # plt.imshow(dMRI_data[:,15,:,cnt].transpose(1,0),cmap='Greys')
        plt.savefig(str(cnt)+'.png',bbox_inches='tight',dpi=300, transparent = True, pad_inches = 0)


    # breakpoint()
    bval = PATH + "bvals"
    bvec = PATH + "bvecs"

    radial_order = 6
    zeta = 700
    lambdaN = 1e-8
    lambdaL = 1e-8
    
    gtab = gradient_table(bvals = bval, bvecs = bvec)
    asm = ShoreModel(gtab, radial_order=radial_order,
                     zeta=zeta, lambdaN=lambdaN, lambdaL=lambdaL)
    asmfit = asm.fit(dMRI_data, mask = mask)
    sphere = get_sphere('symmetric362')
    dMRI_odf = asmfit.odf(sphere)
    dMRI_odf[dMRI_odf<=0]=0

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(dMRI_data, mask)
    dMRI_dti = tenfit.quadratic_form

    FA = fractional_anisotropy(tenfit.evals)
    FA[np.isnan(FA)]=0
    FA = np.clip(FA, 0, 1)
    RGB = color_fa(FA, tenfit.evecs)

    evals = tenfit.evals + 1e-20
    evecs = tenfit.evecs
    cfa = RGB + 1e-20
    cfa /= cfa.max()

    evals = np.expand_dims(evals, 2)
    evecs = np.expand_dims(evecs,2)
    cfa = np.expand_dims(cfa,2)

    ren = window.Scene()
    sphere = get_sphere('symmetric362')
    ren.add(actor.tensor_slicer(evals, evecs, scalar_colors=cfa, sphere=sphere, scale=0.5))
    window.record(ren, n_frames=1, out_path='../data/tensor.png', size=(5000, 5000))


    odf_ = dMRI_odf

    ren = window.Scene()
    sfu = actor.odf_slicer(np.expand_dims(odf_,2), sphere = sphere, colormap = "plasma",scale = 0.5)

    ren.add(sfu)
    window.record(ren, n_frames=1,out_path='../data/odfs.png',size = (5000,5000))



    return None



def main():
    pth = '../data/HCP/996782/T1w/Diffusion/' # Path to the raw diffusion input data

    dMRI2ODF_DTI(pth)


if __name__ == '__main__':
    main()

