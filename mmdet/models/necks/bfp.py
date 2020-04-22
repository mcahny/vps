import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..plugins import NonLocal2D, GeneralizedAttention, AsyncNonLocal2D, GlobalNonLocal2D
from ..registry import NECKS
from ..utils import ConvModule
import pdb

@NECKS.register_module
class BFP(nn.Module):
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
                 refine_level=2,
                 refine_type=None,
                 stack_type='add',
                 conv_cfg=None,
                 norm_cfg=None):
        super(BFP, self).__init__()

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        self.stack_type = stack_type
        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'async_non_local':
            self.refine = AsyncNonLocal2D(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'general_attention':
            self.refine = GeneralizedAttention(
                self.in_channels,
                num_heads=8,
                attention_type='1100')

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


    def forward(self, inputs, inputs_ref=None):
        assert len(inputs) == self.num_levels
        assert isinstance(inputs_ref, list) or inputs_ref is None
        # step1: gather multi-level features by resize and average
        if inputs_ref is not None:
            inputs_list = [inputs]+inputs_ref
        else:
            inputs_list = [inputs]

        bsf_list = []
        for inputs_ in inputs_list:
            feats = []
            gather_size = inputs_[self.refine_level].size()[2:]
            for i in range(self.num_levels):
                if i < self.refine_level:
                    gathered = F.adaptive_max_pool2d(
                        inputs_[i], output_size=gather_size)
                else:
                    gathered = F.interpolate(
                        inputs_[i], size=gather_size, mode='nearest')
                feats.append(gathered)
            bsf = sum(feats) / len(feats)
            bsf_list.append(bsf)

        # step 2: refine gathered features
        # if self.refine_type is not None:
        if self.refine_type == 'non_local':
            bsf = self.refine(bsf_list[0])
        elif self.refine_type == 'async_non_local':
            bsf = self.refine(bsf_list[0], bsf_list[1:])

        # step 3: scatter refined features to multi-levels by residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)

            if self.stack_type == 'cat':
                outs.append(torch.cat([residual, inputs[i]], dim=1))
            elif self.stack_type == 'add':
                outs.append(residual + inputs[i])
            else:
                outs.append(residual + inputs[i])

        return tuple(outs)



    def balanced_gather(self, inputs):
        assert len(inputs) == self.num_levels

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


    def forward_bck(self, inputs, inputs_ref=None):
        # step 1: gather multi-level features by resize and average 
        bsf = self.balanced_gather(inputs)
        
        if inputs_ref is not None:
            bsfs_ref = [self.balanced_gather(in_ref) 
                            for in_ref in inputs_ref]
        
        # step 2: refine gathered features
        if self.refine_type is not None:
            if self.refine_type in ['async_non_local', 'global_non_local']:
                bsf = self.refine(bsf, bsfs_ref)
            else:
                bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)

            if self.stack_type == 'cat':
                outs.append(torch.cat([residual, inputs[i]], dim=1))
            else:
                outs.append(residual + inputs[i])

        return tuple(outs)


