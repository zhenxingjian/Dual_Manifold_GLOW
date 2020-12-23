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
                        default=32,
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

    optimizer = torch.optim.Adam(
        Model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.001,
        amsgrad=True)

    print('# params', sum(x.numel() for x in Model.parameters()))

    # load the pre-trained checkpoint
    # can be commented out to train from scratch
    checkpoints = '../models/DTI2ODF-3D-pretrain.pth'
    ckpts = torch.load(checkpoints)
    Model.load_state_dict(ckpts['model_state_dict'])
    optimizer.load_state_dict(ckpts['optimizer_state_dict'])
    del ckpts

    ####################################################################################################################

    best_err = args.best_err
    now_best_err = 1000
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.num_epochs):
            b = 0
            
            Model.train()
            for data in train_dataloader:  
                dti, eig, odf, msk = data
                odf = odf.unsqueeze(1).cuda()  # [B, 1, 32, 32, 32, 362]
                dti = dti.unsqueeze(1).cuda()  # [B, 1, 32, 32, 32, 3, 3]
                eig = eig.unsqueeze(1).cuda()  # [B, 1, 32, 32, 32, 3]
                msk = msk.unsqueeze(1).cuda()

                logdet, logpz, odff, dtii, eigg = Model(odf, dti, eig)

                loss = logdet + logpz * 0.001  # NOTE
                loss = - loss.mean()
                odf_, dti_, eig_ = Model(odff, dtii, eigg, extra = 8, reverse=True)
               
                edti = (msk.unsqueeze(-1).unsqueeze(-1) * ((dti - dti_) ** 2)).sum()
                eodf = (msk.unsqueeze(-1) * ((odf - odf_) ** 2)).sum()
                eeig = (msk.unsqueeze(-1) * ((eig - eig_) ** 2)).sum()

                logger.info(
                    "Epoch [{}/{}], Iter [{}] Loss: {:.3f}  ldet: {:.3f}  lz: {:.3f} reconstruct: dti {:.3f} odf {:.3f} eig {:.3f}".format(
                        epoch + 1, args.num_epochs, b + 1, loss.item(), -logdet.mean().item(), -logpz.mean().item(), edti.item(), eodf.item(), eeig.item()
                    ))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                b += 1
            
            Model.eval()
            errs = 0
            count = 0
            with torch.no_grad():
                for data in train_dataloader:
                    dti, eig, odf, msk = data
                    odf = odf.unsqueeze(1).cuda()  # [B, 1, 32, 32, 32, 362]
                    dti = dti.unsqueeze(1).cuda()  # [B, 1, 32, 32, 32, 3, 3]
                    eig = eig.unsqueeze(1).cuda()  # [B, 1, 32, 32, 32, 3]
                    msk = msk.unsqueeze(1).cuda()

                    rodf = Model(dti, eig, None, reconstruct = True)

                    err = (msk.unsqueeze(-1)*(rodf - odf) ** 2).sum() / (msk.sum())  # NOTE hard code
                    err = err.sum()
                    errs += err
                errs = errs / len(train_dataset)
                logger.info(
                    "Epoch [{0}/{1}], Reconstruction Loss: {2:.3f}".format(
                        epoch + 1, args.num_epochs, errs.item()
                    ))

            if errs < now_best_err:
                now_best_err = errs

            if errs < best_err:
                best_err = errs

                torch.save({
                    'model_state_dict': Model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, '../models/{}-{}.pth'.format(model.info, errs.item()))


if __name__ == '__main__':
    main()

    







