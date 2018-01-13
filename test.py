import os
import sys

import nibabel as nib 
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vox_resnet import VoxResNet_V1
from dataset import ISLESDataset
from train import GetDataset

def LoadADC(data_root):
    folders = os.listdir(data_root)
    for folder in folders:
        data_type = folder.split('.')[4]
        if data_type != 'MR_ADC':
            continue
        nii_file = os.path.join(data_root, folder, folder+'.nii')
        nii_data = nib.load(nii_file)
        return nii_data
    return None

def GetSMIR_ID(data_root):
    folders = os.listdir(data_root)
    for folder in folders:
        data_type = folder.split('.')[4]
        if data_type != 'MR_MTT':
            continue
        return folder.split('.')[5]
    return None

def Cvt2Nii(predict, folder):
    # refer to https://www.smir.ch/Content/scratch/isles/nibabel_copy_header.py
    predict = predict.astype(np.uint8)
    nii_data = LoadADC(folder)
    #print(nii_data.header)
    #print(nii_data.shape, predict.shape)
    nii_data.set_data_dtype(np.dtype(np.uint8))
    nii_data.get_data()[...] = predict
    #nii_data.get_data()[:] = 1
    #print(nii_data.header)
    return nii_data

def Evaluate(net, dataset, use_cuda):
    net.eval()
    if use_cuda:
        net.cuda()
    dataset.eval()
    for i, (volume, _) in enumerate(dataset):
        folder = dataset.folders[i]
        print('processing %s' % folder)
        volume = Variable(volume)
        if use_cuda:
            volume = volume.cuda()
        predict = net(volume.unsqueeze(0))
        #predict = predict.data.squeeze().permute(2,1,0) # D,H,W -> W,H,D 
        predict = predict.data.squeeze().permute(1,2,0) # D,H,W -> H,W,D 
        predict = predict>0 # 0 for background, 1 for foreground
        predict = predict.cpu().numpy()
        nii_data = Cvt2Nii(predict, folder)
        nib.save(nii_data, './result/SMIR.3DCNN.%s.nii' % GetSMIR_ID(folder))
        predict = nii_data.get_data()
        '''
        for i in range(predict.shape[2]):
        	pred = predict[:, :, i].astype(np.uint8)
        	cv2.imshow('pred', pred*255)
        	cv2.waitKey()
        '''
        

def GetTestData():
    train_dataset,_ = GetDataset()
    data_root = './data/test'
    folders = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root)) ] 
    test_dataset = ISLESDataset(folders, means=train_dataset.means, 
        norm=train_dataset.norm, is_train=False)
    return test_dataset

def GetModel():
    net = VoxResNet_V1(7, 1)
    net.load_state_dict(torch.load('./model/epoch_1600.pt'))
    return net

if __name__ == '__main__':
    train_dataset, val_dataset = GetDataset()
    #test_dataset = GetTestData()
    #test_dataset.eval()
    net = GetModel()
    Evaluate(net, train_dataset, True)
    Evaluate(net, val_dataset, True)

