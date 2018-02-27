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
from train_brats import SplitAndForward, GetDataset
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

def Evaluate(net, dataset, output_dir, vis=False):
    net.eval()
    net.cuda()
    dataset.eval()
    for i, (volume, _) in enumerate(dataset):
        folder = dataset.folders[i]
        print('processing %s' % folder)
        volume = Variable(volume, volatile=True).cuda()
        predict = SplitAndForward(net, volume, 31)
        predict = torch.max(predict.squeeze(), dim=1)[1] 
        predict = predict.cpu().numpy()
        # save result
        mha_data = Cvt2Mha(predict, folder)
        mha_file = 'VSD.%s.%s.mha' % (folder.split('/')[-1], GetSMIR_ID(folder))
        sitk.WriteImage(mha_data, os.path.join(output_dir, mha_file))
        if vis:
            predict = mha_data.get_data()
            for i in range(predict.shape[2]):
                pred = predict[:, :, i].astype(np.uint8)
                cv2.imshow('pred', pred*255)
                cv2.waitKey()

def GetTestData(test_set):
    if test_set == 'test':
        data_root = './data/BRATS/test/HGG_LGG/'
        folders = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root)) ]
        test_dataset = BRATSDataset(folders,  is_train=False)
    else:
        test_set = int(test_set)
        _, test_dataset = GetDataset(test_set, num_fold=5, need_train=False, need_val=True)
    return test_dataset

def GetModel(model_file):
    net = RefineNet(4,5)
    print(net.state_dict().keys())
    net.load_state_dict(torch.load(model_file))
    return net

if __name__ == '__main__':
    model_file = sys.argv[1]
    test_set = sys.argv[2]

    net = GetModel(model_file)
    print("load net done.")

    # train_dataset, val_dataset = GetDataset()
    test_dataset = GetTestData(test_set)
    print("get data done.")
    #test_dataset.eval()

    output_dir = os.path.join('./result_BRATS', test_set)
    try:
        os.makedirs(output_dir)
    except:
        pass
    Evaluate(net, test_dataset, output_dir)

