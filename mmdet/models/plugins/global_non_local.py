import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init

from ..utils import ConvModule
from .async_non_local import AsyncNonLocal2D
import pdb

class GlobalNonLocal2D(AsyncNonLocal2D):
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

    def forward(self, x, x_refs):
        # x_refs: list of x_ref
        n, _, h, w = x.shape

        # g_x: [N, HxW, C]
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0,2,1)

        # global: including x itself as reference feature
        x_refs.append(x)

        # h_xs: [N, TxHxW, C]
        h_xs = [self.h(x_ref).view(n, self.inter_channels, -1) 
                for x_ref in x_refs]

        # Global Temporal Space: includes itself.
        h_xs = torch.cat(h_xs, dim=2).permute(0,2,1)

        # theta_x: [N, HxW, C]
        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0,2,1)

        # phi_xs: [N, C, TxHxW]
        phi_xs = [self.phi(x_ref).view(n, self.inter_channels, -1) 
                    for x_ref in x_refs]
        phi_xs = torch.cat(phi_xs, dim=2)

        pairwise_func = getattr(self, self.mode)
        # pairwase_weight: [N, HxW, TxHxW]
        pairwise_weight = pairwise_func(theta_x, phi_xs)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, h_xs)
        # y: [N, C, H, W]
        y = y.permute(0,2,1).reshape(n, self.inter_channels, h, w)

        output = x + self.conv_out(y)

        return output
