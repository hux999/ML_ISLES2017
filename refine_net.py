import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from vox_resnet import VoxResNet

class RCU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RCU, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
            )
        self.adaptive = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.adaptive(self.block(x)+x)

class ChainResPool(nn.Module):
    def __init__(self, in_channels):
        super(ChainResPool, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(x)
        p1 = F.max_pool3d(x, 3, stride=1, padding=1)
        p1 = self.conv1(p1)
        p2 = F.max_pool3d(x, 3, stride=1, padding=1)
        p2 = self.conv2(p2)
        '''
        p3 = F.max_pool3d(x, 3, stride=1, padding=1)
        p3 = self.conv3(p3)
        '''
        return x+p1+p2

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.theta = nn.Conv3d(in_channels, in_channels/2, kernel_size=1)
        self.phi = nn.Conv3d(in_channels, in_channels/2, kernel_size=1)
        self.g = nn.Conv3d(in_channels, in_channels/2, kernel_size=1)
        self.h = nn.Conv3d(in_channels/2, in_channels, kernel_size=1)

    def forward(self, x):
        N, C, D, H, W = x.shape
        x_pooled = F.max_pool3d(x, kernel_size=(1,2,2), stride=(1,2,2)) # max pooling in spatial domain
        x_theta = self.theta(x).view(N, C/2, D*H*W) 
        x_phi = self.phi(x_pooled).view(N, C/2, D*H*W/4)
        x_g = self.g(x_pooled).view(N, C/2, D*H*W/4)
        x_f = F.softmax(torch.matmul(x_phi.transpose(1,2), x_theta), dim=1) # [N, D*H*W/4, D*H*W]
        return self.h(torch.matmul(x_g, x_f).view(N, C/2, D, H, W))  +  x

class RefineNet(VoxResNet):
    def __init__(self, in_channels, num_classes):
        super(RefineNet, self).__init__(in_channels, num_classes, [32,64,128,256])

        ftr_size = 256

        # adaptive 
        self.adaptive1 = nn.Conv3d(32, ftr_size, kernel_size=1)
        self.adaptive2 = nn.Conv3d(64, ftr_size, kernel_size=1)
        self.adaptive3 = nn.Conv3d(128, ftr_size, kernel_size=1)
        self.adaptive4 = nn.Conv3d(256, ftr_size, kernel_size=1)

        # output conv
        self.smooth1 = nn.Conv3d(ftr_size, ftr_size, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv3d(ftr_size, ftr_size, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv3d(ftr_size, ftr_size, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv3d(ftr_size, ftr_size, kernel_size=3, padding=1)

        self.predict = nn.Conv3d(ftr_size, num_classes, kernel_size=1, bias=True)

        self.non_local = NonLocalBlock(256)

    def upsample_3d(self, x, scale_factor):
        n, c, d, h, w = x.size()
        dst_h, dst_w = h*scale_factor, w*scale_factor
        x = x.view(n, c*d, h, w)
        x = F.upsample(x, size=(dst_h, dst_w), mode='bilinear')
        x = x.view(n, c, d, dst_h, dst_w)
        return x

    def forward(self, x):
        h1 = self.foward_stage1(x)
        h2 = self.foward_stage2(h1)
        h3 = self.foward_stage3(h2)
        h4 = self.foward_stage4(h3)
        h4 = self.non_local(h4)

        h1 = self.adaptive1(F.relu(h1, inplace=False))
        h2 = self.adaptive2(F.relu(h2, inplace=False))
        h3 = self.adaptive3(F.relu(h3, inplace=False))
        h4 = self.adaptive4(F.relu(h4, inplace=False))
        
        p4 = h4
        p3 = self.upsample_3d(p4, 2) + h3
        p3 = self.smooth3(p3)
        p2 = self.upsample_3d(p3, 2) + h2
        p2 = self.smooth2(p2)
        p1 = self.upsample_3d(p2, 2) + h1
        p1 = self.smooth1(p1)

        c = self.predict(p1)
        return c

