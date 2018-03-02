import sys
import os
import time

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

def SplitAndForward(net, x, split_size=31):
    predict = []
    for i, sub_x in enumerate(torch.split(x, split_size, dim=1)):
        result = net(sub_x.unsqueeze(0))
        predict.append(result.data)
    predict = torch.cat(predict, dim=2)
    return predict

def Evaluate(net, dataset, data_name):
    net.eval()
    dataset.eval()
    evaluator_complete = [ EvalDiceScore(), EvalSensitivity(), EvalPrecision() ]
    evaluator_core = [ EvalDiceScore(), EvalSensitivity(), EvalPrecision()  ]
    evaluator_enhancing = [ EvalDiceScore(), EvalSensitivity(), EvalPrecision() ]
    total_time = 0
    for i in range(len(dataset)):
        start = time.time()
        volume, label = dataset[i]
        print('processing %d/%d' % (i, len(dataset)))
        volume = Variable(volume, volatile=True).cuda()
        label = label.cuda()
        predict = SplitAndForward(net, volume, 31)
        predict = torch.max(predict, dim=1)[1] 
        end = time.time()
        print('timing %f' % (end-start))
        total_time += end-start
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
    print('avg time %f' % total/(len(dataset)-1))
    eval_dict = {}
    for evaluator in evaluator_core:
        eval_value = evaluator.Eval()
        eval_dict['/'.join([data_name, type(evaluator).__name__, 'core'])] = eval_value
        print('core: %s, %f' % (type(evaluator).__name__, eval_value))
    for evaluator in evaluator_enhancing:
        eval_value = evaluator.Eval()
        eval_dict['/'.join([data_name, type(evaluator).__name__, 'enhancing'])] = eval_value
        print('enhancing: %s, %f' % (type(evaluator).__name__, eval_value))
    for evaluator in evaluator_complete:
        eval_value = evaluator.Eval()
        eval_dict['/'.join([data_name, type(evaluator).__name__, 'whole'])] = eval_value
        print('complete: %s, %f' % (type(evaluator).__name__, eval_value))
    return eval_dict

def Train(train_data, val_data, net, num_epoch, lr, output_dir):
    net = torch.nn.DataParallel(net, device_ids=[0, 1, 2, 3])
    solver = Solver(net, train_data, 0.0001, output_dir)
    solver.criterion = lambda p,t: SegLoss(p, t, num_classes=5)
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
            #print('val')
            eval_dict_val = Evaluate(net, val_data, 'val')
            for key, value in eval_dict_val.items():
                solver.writer.add_scalar(key, value, i_epoch)
            #print('train')
            #eval_dict_train = Evaluate(net, train_data, 'val')

def GetDataset(fold, num_fold, need_train=True, need_val=True):
    data_root = './data/BRATS/train/HGG/'
    folders_HGG = [ os.path.join(data_root, folder) for folder in 
            sorted(os.listdir(data_root)) ] 
    data_root = './data/BRATS/train/LGG/'
    folders_LGG = [ os.path.join(data_root, folder) for folder in
            sorted(os.listdir(data_root)) ]
    train_folders = []
    val_folders = []
    for i, folder in enumerate(folders_HGG +folders_LGG):
        if i % num_fold == fold:
            val_folders.append(folder)
        else:
            train_folders.append(folder)
    #train_folders = folders_HGG[:2] + folders_LGG[:2]
    #val_folders = folders_HGG[-2:] + folders_LGG[-2:]
    if need_train and len(train_folders)>0:
        train_dataset = BRATSDataset(train_folders, is_train=True,
                sample_shape=(128,128,12))
    else:
        train_dataset = None
    if need_val and len(val_folders)>0:
        val_dataset = BRATSDataset(val_folders, is_train=False)
    else:
        val_dataset = None
    return train_dataset, val_dataset

if __name__ == '__main__':
    fold = int(sys.argv[1])
    train_dataset, val_dataset = GetDataset(fold, num_fold=5)
    print('number of training %d' % len(train_dataset))
    if val_dataset is not None:
        print('number of validation %d' % len(val_dataset))
    #net = VoxResNet_V0(4, 5)
    net = RefineNet(4,5)
    #net = VoxResNet_V1(4, 5)

    output_dir = './output/brast_%d' % fold
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

