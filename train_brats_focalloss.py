import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from solver import Solver, SegLoss

from vox_resnet import VoxResNet_V0, VoxResNet_V1
from refine_net import RefineNet
from dataset import ISLESDataset, BRATSDataset
from evaluator import EvalDiceScore, EvalSensitivity, EvalPrecision
from FocalLoss import FocalLoss
from train_brats import Evaluate, GetDataset


def Train(train_data, val_data, net, num_epoch, lr, output_dir):
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    solver = Solver(net, train_data, 0.0001, output_dir)
    solver.criterion = lambda p,t: SegLoss(p, t, num_classes=5, loss_fn=FocalLoss(5))
    solver.iter_per_sample = 100
    for i_epoch in range(0, num_epoch, solver.iter_per_sample):
        # train
        solver.dataset.set_trans_prob(i_epoch/1000.0+0.5)
        loss = solver.step_one_epoch(batch_size=40, iter_size=1)
        i_epoch = solver.num_epoch
        print(('epoch:%d, loss:%f')  % (i_epoch, loss))

        if i_epoch % 100 == 0:
            save_path = solver.save_model()
            print('save model at %s' % save_path)
        
        # val
        if i_epoch % 100 == 0 and val_data is not None:
            #print('val')
            eval_dict_val = Evaluate(net, val_data, 'val')
            for key, value in eval_dict_val.items():
                solver.writer.add_scalar(key, value, i_epoch)
            #print('train')
            #eval_dict_train = Evaluate(net, train_data, 'val')

def GetModel(model_file):
    net = RefineNet(4,5)
    state_dict = torch.load(model_file)
    rename_state_dict = {}
    for key, value in state_dict.items():
        rename_state_dict['.'.join(key.split('.')[1:])] = value
    net.load_state_dict(rename_state_dict)
    return net

if __name__ == '__main__':
    fold = int(sys.argv[1])
    pretrain = sys.argv[2]
    train_dataset, val_dataset = GetDataset(fold, num_fold=5)
    print('number of training %d' % len(train_dataset))
    if val_dataset is not None:
        print('number of validation %d' % len(val_dataset))
    #net = VoxResNet_V0(4, 5)
    net = GetModel(pretrain)
    #net = VoxResNet_V1(4, 5)

    output_dir = './output/brast_focalloss_%d' % fold
    try:
        os.makedirs(os.path.join(output_dir, 'model'))
    except:
        pass
    try:
        os.makedirs(os.path.join(output_dir, 'tensorboard'))
    except:
        pass
    Train(train_dataset, val_dataset, net,
        num_epoch=3000, lr=0.0001, output_dir=output_dir)

