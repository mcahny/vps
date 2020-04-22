import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init

from ..utils import ConvModule
import pdb

class AsyncNonLocal2D(nn.Module):
    """Non-local module.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    Inputs:
        x: normal x_feat
        x_refs: list of x_ref features from reference frames.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian'):
        super(AsyncNonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product']

        # g, h, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.theta = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.phi = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=None)

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        for m in [self.g, self.theta, self.phi]:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1]**-0.5
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x, x_ref):
        # x_ref : reference feature
        n, _, h, w = x.shape
        
        # v_ref: [N, THW, C]
        v_ref = self.g(x_ref).view(n, self.inter_channels, -1) 
        v_ref = v_ref.permute(0,2,1)
        
        # k_query: [N, HW, C]
        k_query = self.theta(x).view(n, self.inter_channels, -1)
        k_query = k_query.permute(0,2,1)

        # k_ref: [N, C, THW]
        k_ref = self.phi(x_ref).view(n, self.inter_channels, -1)

        # Affinity matrix: [N, HW, THW]
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(k_query, k_ref)

        # y: [N, HW, C]
        y = torch.matmul(pairwise_weight, v_ref)
        # y: [N, C, H, W]
        y = y.permute(0,2,1).reshape(n, self.inter_channels, h, w)

        output = x + self.conv_out(y)
        # Passing only the residual
        # pdb.set_trace()
        # output = self.conv_out(y)

        return output
