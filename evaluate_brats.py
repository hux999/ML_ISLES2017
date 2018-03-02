import sys
import os

import numpy as np
import torch

from vox_resnet import VoxResNet_V0, VoxResNet_V1
from refine_net import RefineNet
from dataset import ISLESDataset, BRATSDataset
from evaluator import EvalDiceScore, EvalSensitivity, EvalPrecision
from train_brats import GetDataset, Evaluate


def GetModel(model_file):
    net = RefineNet(4,5)
    state_dict = torch.load(model_file)
    rename_state_dict = {}
    for key, value in state_dict.items():
        rename_state_dict['.'.join(key.split('.')[1:])] = value
    net.load_state_dict(rename_state_dict)
    return net

if __name__ == '__main__':
    model_file = sys.argv[1]
    test_set = sys.argv[2]

    net = GetModel(model_file)
    net.cuda()
    print("load net done.")

    # train_dataset, val_dataset = GetDataset()
    _, test_dataset = GetDataset(int(test_set), num_fold=5, need_train=False,
            need_val=True)
    print("get data done.")
    #test_dataset.eval()

    Evaluate(net, test_dataset, test_set)
