import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from mmdet.ops import DeformConv

class DeformConvWithOffset(nn.Module):
    
    def __init__(self, in_channels, 
                       out_channels, 
                       kernel_size, 
                       stride=1, 
                       padding=0, 
                       dilation=1, 
                       groups=1, 
                       deformable_groups=1, 
                       bias=True):
        super(DeformConvWithOffset, self).__init__()
        self.conv_offset = nn.Conv2d(in_channels, 
                                     kernel_size * kernel_size * 2 * deformable_groups, 
                                     kernel_size=3, 
                                     stride=1, 
                                     padding=1)
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv = DeformConv(in_channels, 
                               out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride,
                               padding=padding, 
                               dilation=dilation, 
                               groups=groups, 
                               deformable_groups=deformable_groups, 
                               bias=False)
    def forward(self, x):
        return self.conv(x, self.conv_offset(x))