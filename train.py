import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from vox_resnet import VoxResNet
from dataset import ISLESDataset
from evaluate import EvalPrecision,EvalRecall

class CollateFn:
    def __init__(self):
        pass

    def __call__(self, batch_data):
        volume_list = []
        label_list = []
        for volume, label in batch_data:
            volume_list.append(volume)
            label_list.append(label)
        return torch.stack(volume_list), torch.stack(label_list)

def SegLoss(predict, label):
    loss = F.binary_cross_entropy_with_logits(predict.squeeze(), label)
    return loss

def Evaluate(net, dataset, use_cuda):
    net.eval()
    dataset.eval()
    evaluators = [ EvalPrecision(), EvalRecall() ]
    for volume, label in dataset:
        volume = Variable(volume)
        if use_cuda:
            volume = volume.cuda()
            label = label.cuda()
        predict = net(volume.unsqueeze(0))
        predict = predict.data>0
        label = label>0
        for evaluator in evaluators:
            evaluator.AddResult(predict, label)
    for evaluator in evaluators:
        print('%s, %f' % (type(evaluator).__name__, evaluator.Eval()))


def Train(train_data, val_data, net, num_epoch=2000, lr=0.01, use_cuda=True):
    if use_cuda is not None:
        net.cuda()
    #net_ = torch.nn.DataParallel(net, device_ids=use_cuda)
    net_ = net
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for i_epoch in range(num_epoch):
        # train
        net_.train()
        train_data.train()
        batch_data = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=4,
                collate_fn=CollateFn())
        for i_batch, (volume, target) in enumerate(batch_data):
            volume = Variable(volume)
            target = Variable(target)
            if use_cuda:
                volume = volume.cuda()
                target = target.cuda()
            # forward
            predict = net_(volume)
            loss = SegLoss(predict, target)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(('epoch:%d, loss:%f')  % (i_epoch, loss.data[0]))

        # save model for each epoch
        #torch.save(net.state_dict(), ('epoch_%d.pt' % i_epoch))

        # test
        if i_epoch % 10 == 0:
            print('val')
            Evaluate(net, val_data, use_cuda)
            print('train')
            Evaluate(net, train_data, use_cuda)


def GetDataset():
    data_root = './data/train'
    folders = [ os.path.join(data_root, folder) for folder in sorted(os.listdir(data_root)) ] 
    train_dataset = ISLESDataset(folders[:40], is_train=True)
    val_dataset = ISLESDataset(folders[40:], means=train_dataset.means, 
        norm=train_dataset.norm, is_train=False)
    return train_dataset, val_dataset

if __name__ == '__main__':
    train_dataset, val_dataset = GetDataset()
    net = VoxResNet(6, 1)

    Train(train_dataset, val_dataset, net,
        num_epoch=2000, lr=0.0001, use_cuda=True)
