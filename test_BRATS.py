import os
import sys
import time
import pickle
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

import SimpleITK as sitk
import cv2
import numpy as np
import math

from vox_resnet import VoxResNet_V1
from refine_net import RefineNet
from dataset import BRATSDataset, DrawLabel
from train_brats import SplitAndForward, GetDataset

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

def PredictWorker(net, volume, cuda_id, result, lock):
    net.eval()
    net.cuda(cuda_id)
    volume = volume.cuda(cuda_id)
    with torch.no_grad():
        predict = SplitAndForward(net, volume, 31)
        predict = F.softmax(predict.squeeze(), dim=0)
    with lock:
        result[cuda_id] = predict.detach()

def Evaluate(nets, dataset, output_dir):
    dataset.eval()
    for i, (volume, _) in enumerate(dataset):
        folder = dataset.folders[i]
        print('processing %s' % folder)
        lock = threading.Lock()
        result = {}
        if len(nets) > 1:
            threads = [ threading.Thread(target=PredictWorker, args=(net, volume, i, result, lock))
                    for i, net in enumerate(nets) ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            predict = result[0]
            for i in range(1, len(nets)):
                predict += result[i].cuda(0)
        else:
            PredictWorker(nets[0], volume, 0, result, lock)
            predict = result[0]
        predict = torch.max(predict, dim=0)[1] 
        predict = predict.cpu().numpy()
        print(predict.shape)
        # save result
        if output_dir is not None:
            mha_data = Cvt2Mha(predict, folder)
            mha_file = 'VSD.%s.%s.mha' % (folder.split('/')[-1], GetSMIR_ID(folder))
            sitk.WriteImage(mha_data, os.path.join(output_dir, mha_file))
        else:
            for i in range(predict.shape[0]):
                pred = DrawLabel(predict[i, :, :], 4)
                cv2.imwrite('./image/test/%03d_pred.jpg' % i, pred)
                cv2.imshow('pred', pred)
                cv2.waitKey(50)

def GetTestData(test_set):
    if test_set == 'test':
        # testing data
        data_root = './data/BRATS/test/HGG_LGG/'
        folders = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root)) ]
        test_dataset = BRATSDataset(folders,  is_train=False)
    elif test_set.isdigit():
        # training data
        test_set = int(test_set)
        _, test_dataset = GetDataset(test_set, num_fold=5, need_train=False, need_val=True)
    else:
        # single data
        test_dataset = BRATSDataset([test_set],  is_train=False)
    return test_dataset

def GetModel(model_file):
    net = RefineNet(4,5)
    state_dict = torch.load(model_file)
    rename_state_dict = {}
    for key, value in state_dict.items():
        rename_state_dict['.'.join(key.split('.')[1:])] = value
    net.load_state_dict(rename_state_dict)
    return net

if __name__ == '__main__':
    model_files = sys.argv[1]
    test_set = sys.argv[2]

    if os.path.isdir(model_files):
        nets = [ GetModel(os.path.join(model_files, model_file))
                for model_file in  os.listdir(model_files) ]
    else:
        nets = [GetModel(model_file) for model_file in model_files.split(',')]
    print("load net done.")

    # train_dataset, val_dataset = GetDataset()
    test_dataset = GetTestData(test_set)
    print("get data done.")
    #test_dataset.eval()

    if test_set.isdigit() or test_set == 'test':
        output_dir = os.path.join('./result_BRATS', test_set)
        try:
            os.makedirs(output_dir)
        except:
            pass
    else:
        output_dir = None 
    Evaluate(nets, test_dataset, output_dir)

