import sys
import os

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from vox_resnet import VoxResNet_V0, VoxResNet_V1
from refine_net import RefineNet
from dataset import ISLESDataset, BRATSDataset
from evaluator import EvalDiceScore, EvalSensitivity, EvalPrecision
from train_brats import GetDataset

'''
def GetDataset():
    data_root = './data/BRATS/train/HGG/'
    folders_HGG = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root)) ] 
    data_root = './data/BRATS/train/LGG/'
    folders_LGG = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root)) ]
    train_folders = folders_HGG[:20] + folders_LGG[:20]
    train_dataset = BRATSDataset(train_folders, is_train=True, sample_shape=(96,96,7))
    val_folders = folders_HGG[-5:] + folders_LGG[-5:]
    val_dataset = BRATSDataset(val_folders, means=train_dataset.means, 
        norm=train_dataset.norm, is_train=False)
    return train_dataset, val_dataset
'''

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
    evaluator_complete = [ EvalDiceScore(), EvalSensitivity(), EvalPrecision() ]
    evaluator_core = [ EvalDiceScore(), EvalSensitivity(), EvalPrecision()  ]
    evaluator_enhancing = [ EvalDiceScore(), EvalSensitivity(), EvalPrecision() ]
    for volume, label in dataset:
        volume = Variable(volume, volatile=True)
        if use_cuda:
            volume = volume.cuda()
            label = label.cuda()
        #predict = net(volume.unsqueeze(0))
        predict = SplitAndForward(net, volume, 20)
        predict = torch.max(predict, dim=1)[1] 
        predict = predict.long()
        label = label.long()
        # core
        predict_core = torch.min(predict > 0, predict != 2)
        label_core = torch.min(label > 0, label != 2)
        for evaluator in evaluator_core:
            evaluator.AddResult(predict_core, label_core)
        # enhancing
        predict_enhancing = predict == 4
        label_enhancing = label == 4
        for evaluator in evaluator_enhancing:
            evaluator.AddResult(predict_enhancing, label_enhancing)
        # complete
        predict_complete = predict > 0
        label_complete = label > 0
        for evaluator in evaluator_complete:
            evaluator.AddResult(predict_complete, label_complete)
    for evaluator in evaluator_core:
        eval_value = evaluator.Eval()
        print('core: %s, %f' % (type(evaluator).__name__, eval_value))
    for evaluator in evaluator_enhancing:
        eval_value = evaluator.Eval()
        print('enhancing: %s, %f' % (type(evaluator).__name__, eval_value))
    for evaluator in evaluator_complete:
        eval_value = evaluator.Eval()
        print('complete: %s, %f' % (type(evaluator).__name__, eval_value))
    return None

if __name__ == '__main__':
    train_dataset, val_dataset = GetDataset()
    #net = VoxResNet_V1(7, 2)
    #net.load_state_dict(torch.load('./model/max_fscore.pt'))
    net = RefineNet(4,5)
    net.load_state_dict(torch.load('./model/epoch_2400.pt'))
    #net = VoxResNet_V0(7, 2)
    #net.load_state_dict(torch.load('./model/voxresnet/max_fscore.pt'))

    net.cuda()
    Evaluate(net, val_dataset, True)
