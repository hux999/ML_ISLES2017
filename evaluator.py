import torch
import numpy as np

#from metrics import *

def get_tp(gt_data, pred_data):
    return torch.sum(gt_data[pred_data==1])

def get_tn(gt_data, pred_data):
    return torch.sum(gt_data[pred_data==1]==0)

def get_fp(gt_data, pred_data, tp=None):
    if tp is None:
        tp = get_tp(gt_data, pred_data)
    return torch.sum(pred_data) - tp

def get_fn(gt_data, pred_data, tp=None):
    if tp is None:
        tp = get_tp(gt_data, pred_data)
    return torch.sum(gt_data) - tp

class EvalPrecision(object):
    def __init__(self):
        self.tp = 0
        self.tn = 0

    def AddResult(self, predict, target):
        self.tp += get_tp(target, predict)
        self.tn += get_tn(target, predict)

    def Eval(self):
        precision = 1.0 * self.tp / (self.tp+self.tn+1)
        return precision

class EvalRecall(object):
    def __init__(self):
        self.num_hit = 0
        self.num_target = 0
        self.tp = 0
        self.fn = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fn = get_fn(target, predict, tp)
        self.tp += tp
        self.fn += fn

    def Eval(self):
        recall = 1.0 * self.tp / (self.tp+self.fn+1)
        return recall

class EvalFscore(object):
    def __init__(self):
        pass

    def AddResult(self, predict, target):
        pass

    def Eval(self):
        pass

class EvalDiceScore(object):
    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        self.tp += tp
        self.fp += get_fp(target, predict, tp)
        self.fn += get_fn(target, predict, tp)

    def Eval(self):
        return self.tp * 2.0 / (self.tp * 2.0 + self.fn + self.fp + 1.0)


class EvalSensitivity(object):
    def __init__(self):
        self.num_hit = 0
        self.num_target = 0
        self.tp = 0
        self.fn = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fn = get_fn(target, predict, tp)
        self.tp += tp
        self.fn += fn

    def Eval(self):
        recall = 1.0*self.tp/(self.tp+self.fn+1)
        return recall

class EvalHD(object):
    def __init__(self):
        pass

    def AddResult(self, predict, target):
        pass

    def Eval(self):
        pass

if __name__ == '__main__':
    pass
