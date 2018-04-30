import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from tensorboardX import SummaryWriter

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

def SegLoss(predict, label, num_classes=2, loss_fn=nn.CrossEntropyLoss()):
    predict = predict.permute(0, 2, 3, 4, 1).contiguous()
    predict = predict.view(-1, num_classes)
    label = label.view(-1)
    loss = loss_fn(predict, label.long())
    return loss

class Solver(object):

    def __init__(self, net, dataset, lr, output_dir):
        self.net = net
        self.dataset = dataset
        self.output_dir = output_dir
        self.optimizer = self.create_optimizer(lr)
        self.criterion = None # set outside
        self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        self.num_iter = 0
        self.num_epoch = 0
        self.iter_per_sample = 1

    def step_one_epoch(self, batch_size, iter_size=1):
        self.net.cuda()
        self.net.train()
        self.dataset.train()
        self.dataset.set_iter_per_sample(self.iter_per_sample)
        batch_data = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, 
                num_workers=batch_size/2, collate_fn=CollateFn(), pin_memory=True)
        for i_batch, (volume, target) in enumerate(batch_data):
            self.num_iter += batch_size 
            volume = Variable(volume).cuda()
            target = Variable(target).cuda()
            # forward
            predict = self.net(volume)
            loss = self.criterion(predict, target)
            self.writer.add_scalar('loss', loss.item(), self.num_iter)
            # backward
            loss.backward()
            if i_batch % iter_size == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        self.writer.file_writer.flush()
        self.num_epoch += self.iter_per_sample
        return loss.item()

    def create_optimizer(self, lr):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        return optimizer

    def save_model(self):
        model_name = 'epoch_%04d.pt' % (self.num_epoch)
        save_path = os.path.join(self.output_dir, 'model', model_name)
        torch.save(self.net.state_dict(), save_path)
        return save_path

