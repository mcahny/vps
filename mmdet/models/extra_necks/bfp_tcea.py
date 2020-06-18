import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from ..registry import EXTRA_NECKS
from ..utils import ConvModule, TCEA_Fusion
from ..flow_modules import (WarpingLayer, LiteFlowNetCorr)
from ..flow_modules.resample2d_package.resample2d import Resample2d
from mmdet.datasets.pipelines.flow_utils import vis_flow
import pdb

@EXTRA_NECKS.register_module
class BFPTcea(nn.Module):
    """BFP (Balanced Feature Pyrmamids)

    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    https://arxiv.org/pdf/1904.02701.pdf for details.

    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=1,
                 refine_type=None,
                 nframes=3,
                 center=None,
                 stack_type='add',
                 conv_cfg=None,
                 norm_cfg=None):
        super(BFPTcea, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        self.stack_type = stack_type

        self.nframes = nframes
        self.center = center
        assert 0 <= self.refine_level < self.num_levels

        # liteflownet
        class Object():
            pass
        flow_args = Object()
        flow_args.search_range=4
        self.liteflownet = LiteFlowNetCorr(flow_args, self.in_channels+2)
        self.tcea_fusion = TCEA_Fusion(nf=self.in_channels,
                                       nframes=self.nframes,
                                       center=self.center)
        self.flow_warping = WarpingLayer()

        # if self.refine_type == 'conv':
        assert self.refine_type == 'conv'
        self.refine = ConvModule(
            self.in_channels,
            self.in_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def gather(self, inputs):
        # gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)
        bsf = sum(feats) / len(feats)
        return bsf

    def forward(self, inputs, ref_inputs, flow_init, 
                              next_inputs=None, next_flow_init=None):
        assert len(inputs) == self.num_levels
        # inputs: B,C,H,W
        # Gather multi-level features by resize and average
        bsf = self.gather(inputs)
        ref_bsf = self.gather(ref_inputs)
        B,C,H,W = bsf.size()

        warp_bsf = self.flow_warping(ref_bsf, flow_init)
        flow_fine = self.liteflownet(bsf, warp_bsf, flow_init)
        warp_bsf = self.flow_warping(warp_bsf, flow_fine)

        if next_inputs is not None:
            next_bsf = self.gather(next_inputs)
            next_warp_bsf = self.flow_warping(next_bsf, next_flow_init)
            next_flow_fine = self.liteflownet(bsf, next_warp_bsf, next_flow_init)
            next_warp_bsf = self.flow_warping(next_warp_bsf, next_flow_fine)
            bsf_stack = torch.stack([warp_bsf, bsf, next_warp_bsf], dim=1)
            # B,3,C,H,W
        else:
            bsf_stack = torch.stack([bsf, warp_bsf], dim=1)
        # B,2,C,H,W
        bsf = self.tcea_fusion(bsf_stack)

        # Refinement
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # Scatter refined features to multi-levels by residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            outs.append(residual + inputs[i])
        # return tuple(outs), flow_fine
        return tuple(outs)

