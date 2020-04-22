import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
import sys
from time import time
from .correlation_package.correlation import Correlation
from .resample2d_package.resample2d import Resample2d
from ..utils import ConvModule
import pdb


# class OpticalFlowEstimatorCorr(nn.Module):
#     def __init__(self, in_ch):
#         super(OpticalFlowEstimatorCorr, self).__init__()
#         # self.args = args
#         self.convs = nn.Sequential(
#             ConvModule(in_ch, 128, 3, padding=1, activation='leaky_relu'),
#             ConvModule(128, 128, 3, padding=1, activation='leaky_relu'),
#             ConvModule(128, 96, 3, padding=1, activation='leaky_relu'),
#             ConvModule(96, 64, 3, padding=1, activation='leaky_relu'),
#             ConvModule(64, 32, 3, padding=1, activation='leaky_relu'),
#             ConvModule(32, 2, 3, padding=1, activation=None)
#             )
    
#     def forward(self, x):
#         return self.convs(x)

def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1):
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                          stride=stride, dilation=dilation, 
                          padding=((kernel_size - 1) * dilation) // 2, bias=True),
                nn.LeakyReLU(0.1, inplace=True))


class OpticalFlowEstimatorCorr(nn.Module):

    def __init__(self, ch_in):
        super(OpticalFlowEstimatorCorr, self).__init__()
        self.convs = nn.Sequential(
            conv(ch_in, 64),
            conv(64, 64),
            conv(64, 32),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size = 3, stride = 1, padding = 1, dilation=1, groups=1, bias = True)
        )
    def forward(self, x):
        return self.convs(x)

class LiteFlowNetCorr(nn.Module):
    def __init__(self, args, in_ch):
        super(LiteFlowNetCorr, self).__init__()
        self.args = args
        self.corr = Correlation(pad_size=args.search_range, kernel_size = 1,
                max_displacement=args.search_range, stride1 = 1, 
                stride2 = 1, corr_multiply = 1).cuda()
        self.flow_estimator = OpticalFlowEstimatorCorr(
                in_ch+(args.search_range*2+1)**2)
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x1, x2, flow_init=None):
        corr = self.corr(x1.contiguous(), x2.contiguous())
        if flow_init is not None:
            flow = self.flow_estimator(torch.cat([x1, corr, flow_init],
                    dim=1)) 
        else:
            flow = self.flow_estimator(torch.cat([x1, corr], dim = 1))
        return flow


class MaskEstimator(nn.Module):
    def __init__(self, ch_in):
        super(MaskEstimator, self).__init__()
        # self.args = args
        self.convs = nn.Sequential(
            ConvModule(ch_in, ch_in//2, 3, padding=1, activation='leaky_relu'),
            ConvModule(ch_in//2, ch_in//2, 3, padding=1, activation='leaky_relu'),
            ConvModule(ch_in//2, 1, 3, padding=1, activation=None),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        return self.convs(x)


class ImgWarpingLayer(nn.Module):
    def __init__(self):
        super(ImgWarpingLayer, self).__init__()
        self.flow_warping = Resample2d().cuda()
    def forward(self, x, flow):
        return self.flow_warping(x, flow)

# class WarpingLayer(nn.Module):
#     def __init__(self, interp_mode, padding_mode):
#         super(WarpingLayer, self).__init__()
#         self.interp_mode=interp_mode
#         self.padding_mode=padding_mode

#     def forward(self, x, flow):
#         flow = flow.permute(0,2,3,1)
#         assert x.size()[-2:] == flow.size()[1:3]
#         B, C, H, W = x.size()
#         # mesh grid
#         grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
#         grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
#         grid.requires_grad = False
#         grid = grid.type_as(x)
#         vgrid = grid + flow
#         # scale grid to [-1,1]
#         vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
#         vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
#         vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
#         output = F.grid_sample(x, vgrid_scaled, 
#                                mode=self.interp_mode, 
#                                padding_mode=self.padding_mode)
#         return output

class WarpingLayer(nn.Module):
    
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def get_grid(self, x):
        torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
        torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
        grid = torch.cat([torchHorizontal, torchVertical], 1)
        return grid
    
    def forward(self, x, flow):

        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros_like(flow).cuda()
        flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)

        grid = (self.get_grid(x).cuda() + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp




class SELayer(nn.Module):
    def __init__(self, ch_in):
        super(SELayer, self).__init__()
        self.convs = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(ch_in*2, ch_in, 1, padding=0),
            ConvModule(ch_in, ch_in, 1, padding=0),
            ConvModule(ch_in, ch_in, 1, padding=0, activation=None),
            nn.Sigmoid()
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, x):
        return self.convs(x)


if __name__=='__main__':
    
    import matplotlib.pyplot as plt
    class Object(object):
        pass
    args = Object()
    args.corr = 'none'
    args.search_range = 4
    args.batch_norm = False

    flownet = LiteFlowNet(args)
    warping = WarpingLayer()

    x1 = torch.Tensor(1,3,256,256)
    x1.fill_(0)
    x2 = torch.Tensor(1,3,256,256)
    x2.fill_(1)

    flow = flownet(x1,x2)
    x2_warp = warping(x2,flow)
    pdb.set_trace()
