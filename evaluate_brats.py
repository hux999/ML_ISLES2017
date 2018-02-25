import sys
import os

import numpy as np
import torch

from vox_resnet import VoxResNet_V0, VoxResNet_V1
from refine_net import RefineNet
from dataset import ISLESDataset, BRATSDataset
from evaluator import EvalDiceScore, EvalSensitivity, EvalPrecision
from train_brats import GetDataset, Evalute


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
