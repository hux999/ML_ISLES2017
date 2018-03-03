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
from dataset import ISLESDataset
from evaluator import EvalDiceScore, EvalSensitivity, EvalPrecision

from FocalLoss import FocalLoss


def Evaluate(net, dataset, data_name):
    net.eval()
    dataset.eval()
    evaluators = [ EvalDiceScore(), EvalSensitivity(), EvalPrecision() ]
    for volume, label in dataset:
        volume = Variable(volume).cuda()
        label = label.cuda()
        predict = net(volume.unsqueeze(0))
        predict = torch.max(predict, dim=1)[1] 
        predict = predict.data.long()
        label = label.long()
        for evaluator in evaluators:
            evaluator.AddResult(predict, label)
    eval_dict = {}
    for evaluator in evaluators:
        eval_value = evaluator.Eval()
        eval_dict['/'.join([data_name, type(evaluator).__name__])] = eval_value
        print('%s, %f' % (type(evaluator).__name__, eval_value))
    return eval_dict

def Train(train_data, val_data, net, num_epoch, lr, output_dir):
    net = torch.nn.DataParallel(net, device_ids=[0, 1])
    solver = Solver(net, train_data, 0.0001, output_dir)
    solver.criterion = lambda p,t: SegLoss(p, t, num_classes=2)
    solver.iter_per_sample = 100
    for i_epoch in range(0, num_epoch, solver.iter_per_sample):
        # train
        solver.dataset.set_trans_prob(i_epoch/1000.0+0.15)
        loss = solver.step_one_epoch(batch_size=40, iter_size=1)
        i_epoch = solver.num_epoch
        print(('epoch:%d, loss:%f')  % (i_epoch, loss))

        if i_epoch % 100 == 0:
            save_path = solver.save_model()
            print('save model at %s' % save_path)
        
        # val
        if i_epoch % 100 == 0:
            print('val')
            eval_dict_val = Evaluate(net, val_data, 'val')
            for key, value in eval_dict_val.items():
                solver.writer.add_scalar(key, value, i_epoch)
            '''
            print('train')
            eval_dict_train = Evaluate(net, train_data, 'train')
            for key, value in eval_dict_train.items():
                solver.writer.add_scalar(key, value, i_epoch)
            '''

def GetDataset(fold, num_fold, need_train=True, need_val=True):
    data_root = './data/ISLES/train'
    folders = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root)) ] 
    train_folders = []
    val_folders = []
    for i, folder in enumerate(folders):
        if i % num_fold == fold:
            val_folders.append(folder)
        else:
            train_folders.append(folder)
    if need_train and len(train_folders)>0:
        train_dataset = ISLESDataset(train_folders, is_train=True,
                sample_shape=(96,96,7))
    else:
        train_dataset = None
    if need_val and len(val_folders)>0:
        val_dataset = ISLESDataset(val_folders, is_train=False)
    else:
        val_dataset = None
    return train_dataset, val_dataset

if __name__ == '__main__':
    fold = int(sys.argv[1])
    train_dataset, val_dataset = GetDataset(fold, num_fold=6)
    print('number of training %d' % len(train_dataset))
    if val_dataset is not None:
        print('number of validation %d' % len(val_dataset))
    net = RefineNet(9, 2, dropout=False)

    output_dir = './output/isles_%d' % fold
    try:
        os.makedirs(os.path.join(output_dir, 'model'))
    except:
        pass
    try:
        os.makedirs(os.path.join(output_dir, 'tensorboard'))
    except:
        pass
    Train(train_dataset, val_dataset, net,
        num_epoch=2000, lr=0.0001, output_dir=output_dir)
