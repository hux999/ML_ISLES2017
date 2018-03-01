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
        self.sum_score = 0
        self.count = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fp = get_fp(target, predict, tp)
        if tp+tp == 0:
            return
        precision = 1.0*tp/(tp+fp)
        self.sum_score += precision
        self.count += 1

    def Eval(self):
        if self.count > 0:
            return self.sum_score/self.count
        else:
            return 0

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
        self.sum_score = 0
        self.count = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fp = get_fp(target, predict, tp)
        fn = get_fn(target, predict, tp)
        if tp*2.0+fn+fp == 0:
            return
        dice_score = tp * 2.0 / (tp * 2.0 + fn + fp)
        self.sum_score += dice_score
        self.count += 1

    def Eval(self):
        if self.count > 0:
            return self.sum_score/self.count
        else:
            return 0

class EvalSensitivity(object):
    def __init__(self):
        self.sum_score = 0
        self.count = 0

    def AddResult(self, predict, target):
        tp = get_tp(target, predict)
        fn = get_fn(target, predict, tp)
        if tp+fn == 0:
            return
        recall = 1.0*tp/(tp+fn)
        self.sum_score += recall
        self.count += 1

    def Eval(self):
        if self.count > 0:
            return self.sum_score/self.count
        else:
            return 0

class EvalHD(object):
    def __init__(self):
        pass

    def AddResult(self, predict, target):
        pass

    def Eval(self):
        pass

if __name__ == '__main__':
    pass
