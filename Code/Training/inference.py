from __future__ import print_function

import os
import torch
import pprint
import argparse
import numpy as np

import torch.backends.cudnn as cudnn
from torch.nn.init import *
from torch.utils.data import DataLoader
from dataset_HCP import Dataset_GLOW, DataLoaderX

from logger import setup_logger

import model as model
import layers
import utils

def main():
    logger = setup_logger('manifold Dual Glow from DTI to ODF')
    parser = argparse.ArgumentParser()
    """path to store input/ output file"""
    parser.add_argument("--dset",
                        type=str,
                        default='../HCP_processed/',
                        required=False)
    """args for training"""
    parser.add_argument("--num-epochs",
                        help="number of epochs",
                        type=int,
                        default=10000,
                        required=False)
    parser.add_argument("--learning-rate", "-lr",
                        help="learning rate of the model",
                        type=float,
                        default=1e-6,
                        required=False)
    parser.add_argument("--batch-size", "-b",
                        help="batch size of training samples",
                        type=int,
                        default=2,
                        required=False)
    parser.add_argument("--best-err", "-err",
                        type=float,
                        default=1000,
                        required=False)

    args = parser.parse_args()
    logger.info("call with args: \n{}".format(pprint.pformat(vars(args))))
    logger.info("model info {}".format(model.info))

    ####################################################################################################################
    """load train and test set"""
    train_dataset = Dataset_GLOW(data_path = args.dset, train = True)
    train_dataloader = DataLoaderX(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = Dataset_GLOW(data_path = args.dset, train = False)
    test_dataloader = DataLoaderX(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print('finish loading the data')

    Model = model.Glow().cuda()
    cudnn.benchmark = True
    Model = torch.nn.DataParallel(Model)
    print('finish init the model')

    print('# params', sum(x.numel() for x in Model.parameters()))


    checkpoints = '../models/DTI2ODF-3D-pretrain.pth'
    ckpts = torch.load(checkpoints)
    Model.load_state_dict(ckpts['model_state_dict'])
    del ckpts

    ####################################################################################################################
            
    Model.eval()
    errs = 0
    dti_ = []
    rodf_ = []
    odf_ = []
    with torch.no_grad():
        for data in test_dataloader:
            dti, eig, odf, msk = data
            odf = odf.unsqueeze(1).cuda()  # [B, 1, 32, 32, 32, 362]
            dti = dti.unsqueeze(1).cuda()  # [B, 1, 32, 32, 32, 3, 3]
            eig = eig.unsqueeze(1).cuda()  # [B, 1, 32, 32, 32, 3]
            msk = msk.unsqueeze(1).cuda()

            rodf = Model(dti, eig, None, reconstruct = True)

            dti_.append(dti.detach().cpu().numpy())
            odf_.append(odf.detach().cpu().numpy())
            rodf_.append((msk.unsqueeze(-1)*rodf).detach().cpu().numpy())

    dti_ = np.concatenate(dti_,0)
    odf_ = np.concatenate(odf_,0)
    rodf_ = np.concatenate(rodf_,0)
    error = (odf_-rodf_)**2

    np.savez('../data/results.npz', dti = dti_, odf =odf_, rodf = rodf_)



if __name__ == '__main__':
    main()

    







