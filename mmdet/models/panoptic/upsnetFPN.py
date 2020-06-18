import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from ..registry import PANOPTIC
from ..utils import DeformConvWithOffset, ConvModule
from mmdet.ops import DeformConv

import torch
import numpy as np
import pycocotools.mask as mask_util
import pdb

@PANOPTIC.register_module
class UPSNetFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_levels,
                 num_things_classes,
                 num_classes,
                 ignore_label,
                 loss_weight,
                 conv_cfg=None,
                 norm_cfg=None):
        super(UPSNetFPN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.num_things_classes = num_things_classes # 19
        self.num_classes = num_classes # 8
        self.num_stuff_classes = num_classes - num_things_classes # 11
        self.ignore_label = ignore_label
        self.loss_weight = loss_weight
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.deform_convs = nn.ModuleList()
        self.deform_convs.append(nn.Sequential( 
            DeformConvWithOffset(self.in_channels, self.in_channels, 
                                 kernel_size=3, padding=1),
            nn.GroupNorm(32, self.in_channels),
            nn.ReLU(inplace=True),
            DeformConvWithOffset(self.in_channels, self.out_channels,
                                 kernel_size=3, padding=1),
            nn.GroupNorm(32, self.out_channels),
            nn.ReLU(inplace=True),
            DeformConvWithOffset(self.out_channels, self.out_channels,
                                 kernel_size=3, padding=1),
            nn.GroupNorm(32, self.out_channels),
            nn.ReLU(inplace=True),            
            ))
        self.conv_pred = ConvModule(self.out_channels * 4, 
                                    self.num_classes, 1, 
                                    padding=0, 
                                    conv_cfg=self.conv_cfg, 
                                    norm_cfg=self.norm_cfg,
                                    activation=None)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')


    def forward(self, inputs):
        assert len(inputs) == self.num_levels
        fpn_px = []
        for i in range(self.num_levels):
            fpn_px.append(self.deform_convs[0](inputs[i]))

        fpn_p2 = fpn_px[0]
        fpn_p3 = F.interpolate(fpn_px[1], None, 2, mode='bilinear', align_corners=False)
        fpn_p4 = F.interpolate(fpn_px[2], None, 4, mode='bilinear', align_corners=False)
        fpn_p5 = F.interpolate(fpn_px[3], None, 8, mode='bilinear', align_corners=False)
        feat = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5], dim=1)

        fcn_score = self.conv_pred(feat)
        fcn_output = self.upsample(fcn_score)
        return fcn_output, fcn_score
        

    def loss(self, segm_pred, segm_label):
        loss = dict()
        loss_segm = F.cross_entropy(segm_pred, segm_label, ignore_index = self.ignore_label)
        loss['loss_segm'] = self.loss_weight * loss_segm
        return loss

    #### Not used.
    def get_semantic_segm(self, segm_feature_pred, ori_shape,
                          img_shape_withoutpad, ignore_map=None):

        # only support 1 batch
        segm_feature_pred = segm_feature_pred[:, :, 0:img_shape_withoutpad[0], 0:img_shape_withoutpad[1]]
        segm_pred_map = F.softmax(segm_feature_pred, 1)
        segm_pred_map = F.interpolate(segm_pred_map, size=ori_shape[0:2], mode='bilinear', align_corners=False)
        segm_pred_map = torch.max(segm_pred_map, 1).indices
        segm_pred_map = segm_pred_map.float()
        segm_pred_map = segm_pred_map[0]

        segm_pred_map = segm_pred_map.cpu().numpy()
        segm_pred_map_unique = np.unique(segm_pred_map).astype(np.int)
        # cls_segms = [[] for _ in range(self.num_classes - 1)]
        cls_segms = [[] for _ in range(self.num_classes)]
        for i in segm_pred_map_unique:
            # for i only within [0,10]
            if i >= self.num_stuff_classes:
                continue
            cls_im_mask = np.zeros((ori_shape[0], ori_shape[1])).astype(np.uint8)
            cls_im_mask[segm_pred_map==i] = 1

            rle = mask_util.encode( np.array(cls_im_mask[:, :, np.newaxis], order='F'))[0]
            # cls_segms[i-1].append(rle)
            cls_segms[i].append(rle)

        return cls_segms, segm_pred_map.astype(np.uint8)