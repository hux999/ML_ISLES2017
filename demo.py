import os
import sys

import cv2
import numpy as np
import nibabel as nib 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vox_resnet import VoxResNet_V1
from train import GetDataset
from test import GetTestData

def LoadADC(data_root):
    folders = os.listdir(data_root)
    for folder in folders:
        data_type = folder.split('.')[4]
        if data_type != 'MR_ADC':
            continue
        nii_file = os.path.join(data_root, folder, folder+'.nii')
        nii_data = nib.load(nii_file)
        return nii_data.get_data()
    return None

def DrawResult1(mask, canvas, color, alpha=0.5):
    canvas = canvas.astype(np.float32)
    canvas *= (255.0/np.max(canvas))
    mask = mask*alpha
    chn_B = canvas*(1.0-mask) + mask*color[0]
    chn_G = canvas*(1.0-mask) + mask*color[1]
    chn_R = canvas*(1.0-mask) + mask*color[2]
    canvas = cv2.merge([canvas*0.5+chn_B*0.5, canvas*0.5+chn_G*0.5, canvas*0.5+chn_R*0.5])
    canvas = canvas.astype(np.uint8)
    return canvas

def DrawResult2(mask, canvas, color, alpha=0.3):
    canvas = canvas.astype(np.float32)
    canvas *= (255.0/np.max(canvas))
    mask = cv2.blur(mask, (3,3))
    heatmap = cv2.applyColorMap(((mask-0.1)/0.9*255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32)
    canvas = cv2.merge([canvas,canvas,canvas])
    mask = cv2.merge([mask, mask, mask])
    canvas = (canvas*(1-alpha)+heatmap*alpha)*(mask>0.1) + canvas*(mask<=0.1)
    return canvas.astype(np.uint8)

def Demo(net, dataset, use_cuda):
    net.eval()
    if use_cuda:
        net.cuda()
    dataset.eval()
    for i, (volume, label) in enumerate(dataset):
        folder = dataset.folders[i]
        print('processing %s' % folder)
        volume = Variable(volume)
        if use_cuda:
            volume = volume.cuda()
        predict = net(volume.unsqueeze(0))
        predict = F.sigmoid(predict.data).squeeze().cpu().numpy()
        depth = predict.shape[0]
        adc_img = LoadADC(folder)
        for j in range(depth):
            canvas = cv2.resize(adc_img[:, :, j].copy(), (0,0), fx=4, fy=4)
            if label is not None:
                gt = label[j, :, :].cpu().numpy().astype(np.float32)
                gt = cv2.resize(gt, (0,0), fx=4, fy=4)
                gt = DrawResult1(gt, canvas, (0,0,255))
                cv2.imshow('gt', gt)
                cv2.imwrite('image/%d_%d_gt.jpg' % (i,j), gt)
            pred = predict[j, :, :].astype(np.float32)
            pred = cv2.resize(pred, (0,0), fx=4, fy=4)
            pred = DrawResult2(pred , canvas, (255,0,0))
            cv2.imshow('pred', pred)
            cv2.imwrite('image/%d_%d_pred.jpg' % (i,j), pred)
            cv2.waitKey()


if __name__ == '__main__':
    train_dataset,val_dataset = GetDataset()
    #test_dataset = GetTestData()

    net = VoxResNet_V1(7, 1)
    net.load_state_dict(torch.load('./model/epoch_1600.pt'))

    Demo(net, val_dataset, True)
