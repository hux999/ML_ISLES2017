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
    ''' base backend '''
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


    def forward(self, x):
        h1 = self.foward_stage1(x)
        h2 = self.foward_stage2(h1)
        h3 = self.foward_stage3(h2)
        h4 = self.foward_stage4(h4)
        return h4

    def foward_stage1(self, x):
        h = self.conv1_1(x)
        return h

    def foward_stage2(self, x):
        h = self.conv1_2(x)
        h = self.voxres2(h)
        h = self.voxres3(h)
        return h

    def foward_stage3(self, x):
        h = self.conv4(x)
        h = self.voxres5(h)
        h = self.voxres6(h)
        return h

    def foward_stage4(self, x):
        h = self.conv7(x)
        h = self.voxres8(h)
        h = self.voxres9(h)
        return h

class VoxResNet_V0(VoxResNet):
    def __init__(self, in_channels, num_classes):
        super(VoxResNet_V0, self).__init__(in_channels, num_classes)
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
        h1 = self.foward_stage1(x)
        h2 = self.foward_stage2(h1)
        h3 = self.foward_stage3(h2)
        h4 = self.foward_stage4(h3)

        c1 = self.head_c1(h1)
        c2 = self.head_c2(h2)
        c3 = self.head_c3(h3)
        c4 = self.head_c4(h4)

        return c1+c2+c3+c4

class RCU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCU, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            )

    def forward(self, x):
        return self.block(x)+x


class VoxResNet_V1(VoxResNet):
    def __init__(self, in_channels, num_classes):
        super(VoxResNet_V1, self).__init__(in_channels, num_classes)
        self.head1 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm3d(64)
            )
        self.head2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
            )
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(64)
            )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(64)
            )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1,2,2), stride=(1,2,2), bias=False),
            nn.BatchNorm3d(64)
            )

        self.predict = nn.Conv3d(64, num_classes, kernel_size=1, bias=True)

    def forward(self, x):
        h1 = self.foward_stage1(x)
        h2 = self.foward_stage2(h1)
        h3 = self.foward_stage3(h2)
        h4 = self.foward_stage4(h3)

        f1 = self.upsample1(h4)
        f2 = self.upsample2(f1+h3)
        f3 = self.upsample3(f2+h2)

        h1 = self.head1(h1)
        f4 = self.head2(f3+h1)
        c = self.predict(f4)
        return c

if __name__ == '__main__':
    net = VoxResNet_V0(87, 2)
    net.cuda()
    while True:
        data = torch.rand(1, 87, 5, 192, 192)
        c = net(Variable(data).cuda())



