import torch

class EvalPrecision(object):
    def __init__(self):
        self.num_hit = 0
        self.num_pred = 0

    def AddResult(self, predict, target):
        self.num_hit += torch.sum( (predict==target) & target)
        self.num_pred += torch.sum(predict)

    def Eval(self):
        if self.num_pred >0:
            precision = 1.0*self.num_hit/self.num_pred
        else:
            precision = -1.0
        return precision

class EvalRecall(object):
    def __init__(self):
        self.num_hit = 0
        self.num_target = 0

    def AddResult(self, predict, target):
        self.num_hit += torch.sum( (predict==target) & target)
        self.num_target += torch.sum(target)

    def Eval(self):
        if self.num_target >0:
            recall = 1.0*self.num_hit/self.num_target
        else:
            recall = -1.0
        return recall

if __name__ == '__main__':
    pass
