import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VoxRex(nn.Module):
    def __init__(self, in_channels):
        super(VoxRex, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x)+x

class VoxResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VoxResNet, self).__init__()
        self.conv1_1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False)
            )
        self.conv1_2 = nn.Sequential(
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1,2,2), padding=1, bias=True)
            )
        self.voxres2 = VoxRex(64)
        self.voxres3 = VoxRex(64)
        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=(1,2,2), padding=1, bias=True)
            )
        self.voxres5 = VoxRex(64)
        self.voxres6 = VoxRex(64)
        self.conv7 = nn.Sequential(
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, stride=(1,2,2), padding=1, bias=True)
            )
        self.voxres8 = VoxRex(64)
        self.voxres9 = VoxRex(64)
        self.head_c1 = nn.Conv3d(32, num_classes, kernel_size=1, bias=True)
        self.head_c2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1, bias=True)
            )
        self.head_c3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1, bias=True)
            )
        self.head_c4 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(64, 32, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, num_classes, kernel_size=1, bias=True)
            )

    def forward(self, x):
        h = self.conv1_1(x)
        c1 = self.head_c1(h)

        h = self.conv1_2(h)
        h = self.voxres2(h)
        h = self.voxres3(h)
        c2 = self.head_c2(h)

        h = self.conv4(h)
        h = self.voxres5(h)
        h = self.voxres6(h)
        c3 = self.head_c3(h)

        h = self.conv7(h)
        h = self.voxres8(h)
        h = self.voxres9(h)
        c4 = self.head_c4(h)

        return c1+c2+c3+c4

if __name__ == '__main__':
    net = VoxResNet(87, 2)
    net.cuda()
    while True:
        data = torch.rand(1, 87, 5, 192, 192)
        c = net(Variable(data).cuda())



