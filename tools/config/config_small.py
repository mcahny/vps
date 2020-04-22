# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project. 
#
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Written by Yuwen Xiong
# --------------------------------------------------------

import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

# Cityscapes dataset by default
config.dataset = edict()
config.dataset.num_classes = 9
config.dataset.num_seg_classes = 19
config.dataset.dataset = "Cityscapes"
config.dataset.dataset_path = ""
config.dataset.image_set = ""
config.dataset.test_image_set = ""

config.network = edict()
config.network.has_rpn = True
config.network.has_rcnn = True
config.network.has_mask_head = True
config.network.has_fcn_head = True
config.network.fcn_head = "FCNHead"
config.network.fcn_num_layers = 2
config.network.has_panoptic_head = True

config.test = edict()
config.test.scales = 0
config.max_size = 0
config.batch_size = 1
config.panoptic_stuff_area_limit = 4096


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                config[k] = v

