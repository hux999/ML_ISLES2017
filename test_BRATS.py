import os
import sys
import time
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import SimpleITK as sitk
import cv2
import numpy as np
import math

from vox_resnet import VoxResNet_V1
from refine_net import RefineNet
from dataset import BRATSDataset
from train import GetDataset
import pdb

def LoadFlair(data_root):
    folders = os.listdir(data_root)
    for folder in folders:
        data_type = folder.split('.')[4]
        if data_type != 'MR_Flair':
            continue
        mha_file = os.path.join(data_root, folder, folder+'.mha')
        mha_data = sitk.ReadImage(mha_file)
        return mha_data
    return None

def GetSMIR_ID(data_root):
    folders = os.listdir(data_root)
    for folder in folders:
        data_type = folder.split('.')[4]
        if data_type != 'MR_Flair':
            continue
        return folder.split('.')[5]
    return None

def Cvt2Mha(predict, folder):
    # refer to https://www.smir.ch/Content/scratch/isles/nibabel_copy_header.py
    predict = predict.astype(np.uint8)
    mha_data = sitk.GetImageFromArray(predict)
    return mha_data

def SplitAndForward(net, x, split_size=5):
    predict = []
    for i, sub_x in enumerate(torch.split(x, split_size, dim=1)):
        result = net(sub_x.unsqueeze(0))
        predict.append(result.data)
    predict = torch.cat(predict, dim=2)
    return predict


def Evaluate(net, dataset, use_cuda):
    net.eval()
    dataset.eval()
    if use_cuda:
        net.cuda()
    dataset.eval()
    for i, (volume, _) in enumerate(dataset):
        folder = dataset.folders[i]
        print('processing %s' % folder)
        volume = Variable(volume, volatile=True)
        if use_cuda:
            volume = volume.cuda()
            net = net.cuda()
        predict = SplitAndForward(net, volume, 20)
        predict = torch.max(predict, dim=1)[1] 
        predict = predict.cpu().numpy()[0]      

        mha_data = Cvt2Mha(predict, folder)
        print folder[-6:]
        sitk.WriteImage(mha_data, './result_BRATSD/Testing/VSD.%s.%s.mha' % (folder[-6:],GetSMIR_ID(folder)))
        # predict = mha_data.get_data()
        '''
        for i in range(predict.shape[2]):
        	pred = predict[:, :, i].astype(np.uint8)
        	cv2.imshow('pred', pred*255)
        	cv2.waitKey()
        '''
        

def GetTestData():
    #for train_dataset
    # data_root = '/home/tinzhuo/ML_ISLES2017/data_BRATS/BRATS2015_Training/HGG'
    #for test_dataset
    data_root = '/home/tinzhuo/ML_ISLES2017/data_BRATS/Testing/HGG_LGG'
    folders = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root)) ] 
    test_dataset = BRATSDataset(folders[:1],  is_train=True, sample_shape=(96,96,5))
    return test_dataset

def GetModel():
    net = RefineNet(4,5)
    net.load_state_dict(torch.load('./model_BRATSD/epoch_2400.pt'))
    return net

if __name__ == '__main__':
    # train_dataset, val_dataset = GetDataset()
    test_dataset = GetTestData()
    print "get data done."
    #test_dataset.eval()
    net = GetModel()
    print "load net done."

    Evaluate(net, test_dataset, True)
    # Evaluate(net, val_dataset, True)

