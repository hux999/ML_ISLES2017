import os
import sys
import time
import threading

import nibabel as nib 
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vox_resnet import VoxResNet_V1
from refine_net import RefineNet
from dataset import ISLESDataset, DrawLabel
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
    predict = predict.astype(np.uint16)
    nii_data = LoadADC(folder)
    nii_data.get_data()[...] = predict
    #nii_data.set_data_dtype(np.uint16)
    return nii_data

def PredictWorker(net, volume, cuda_id, result, lock):
    net.eval()
    net.cuda(cuda_id)
    volume = volume.cuda(cuda_id)
    predict = net(volume.unsqueeze(0))
    predict = F.softmax(predict.squeeze(), dim=0).data
    with lock:
        result[cuda_id] = predict

def Evaluate(nets, dataset, output_dir):
    dataset.eval()
    for i, (volume, _) in enumerate(dataset):
        folder = dataset.folders[i]
        print('processing %s' % folder)
        start = time.time()
        volume = Variable(volume).cuda()
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
                #predict += result[i].cuda(0)
                predict = torch.max(predict, result[i].cuda(0))
        else:
            PredictWorker(nets[0], volume, 0, result, lock)
            predict = result[0]
        predict[1,:,:,:] = predict[1,:,:,:]*1.0
        predict = torch.max(predict, dim=0)[1]
        predict = predict.permute(1,2,0) # D,H,W -> H,W,D 
        predict = predict.cpu().numpy()
        end = time.time()
        print('time', end-start)
        if output_dir is not None:
            nii_data = Cvt2Nii(predict, folder)
            nii_file = 'SMIR.%s.%s.nii' % (folder.split('/')[-1] ,GetSMIR_ID(folder))
            print('save at %s' % nii_file)
            nib.save(nii_data, os.path.join(output_dir, nii_file))
        adc_img = LoadADC(folder).get_data()
        for j in range(predict.shape[2]):
            canvas = adc_img[:, :, j].copy()
            canvas = canvas.astype(np.float32)
            canvas = (canvas*(255.0/np.max(canvas))).astype(np.uint8)
            canvas = cv2.merge([canvas, canvas, canvas])
            pred = DrawLabel(predict[:, :, j], 4)
            pred = np.maximum(canvas, pred)
            cv2.imshow('pred', pred)
            cv2.imshow('adc', canvas)
            output = os.path.join('./image', folder.split('/')[-1])
            try:
                os.makedirs(output)
            except:
                pass
            cv2.imwrite(output+('/%03d.jpg' % j), pred)
            cv2.waitKey()

def GetTestData(test_set):
    if test_set == 'test':
        # testing data
        data_root = './data/ISLES/test/'
        folders = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root)) ]
        test_dataset = ISLESDataset(folders,  is_train=False)
    elif test_set.isdigit():
        # training data
        test_set = int(test_set)
        _, test_dataset = GetDataset(test_set, num_fold=6, need_train=False, need_val=True)
    else:
        # single data
        test_dataset = ISLESDataset([test_set],  is_train=False)
    return test_dataset

def GetModel(model_file):
    net = RefineNet(9,2)
    state_dict = torch.load(model_file)
    rename_state_dict = {}
    for key, value in state_dict.items():
        rename_state_dict['.'.join(key.split('.')[1:])] = value
    net.load_state_dict(rename_state_dict)
    return net

if __name__ == '__main__':
    model_files = sys.argv[1]
    if os.path.isdir(model_files):
        model_files = [ os.path.join(model_files, model) for model in os.listdir(model_files)]
    else:
        model_files = model_files.split(',')
    test_set = sys.argv[2]

    nets = [GetModel(model_file) for model_file in model_files]
    print("load net done.")

    # train_dataset, val_dataset = GetDataset()
    test_dataset = GetTestData(test_set)
    print("get data done.")
    #test_dataset.eval()

    if test_set.isdigit() or test_set == 'test':
        output_dir = os.path.join('./result_isles', test_set)
        try:
            os.makedirs(output_dir)
        except:
            pass
    else:
        output_dir = './result_isles/test'
    Evaluate(nets, test_dataset, output_dir)


