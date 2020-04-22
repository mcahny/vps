# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Modifications Copyright (c) 2019 Uber Technologies, Inc.
# ---------------------------------------------------------------------------
# Based on:
# ---------------------------------------------------------------------------
# Detectron
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------

"""Representation of the standard COCO json dataset format.
When working with a new dataset, we strongly suggest to convert the dataset into
the COCO json format and use the existing code; it is not recommended to write
code to support new dataset formats.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import pickle
import numpy as np
import os
import scipy.sparse

# Must happen before importing COCO API (which imports matplotlib)
"""Set matplotlib up."""
# import matplotlib
# # Use a non-interactive backend
# matplotlib.use('Agg')
# COCO API
from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

from tools.config.config import config

# from lib.utils.timer import Timer
# import upsnet.bbox.bbox_transform as box_utils
# import upsnet.mask.mask_transform as segm_utils

# from lib.utils.logging import logger
import pdb

class JsonDataset(object):
    """A class representing a COCO json dataset."""

    def __init__(self, name, image_dir, anno_file):
        # if logger:
        #     logger.info('Creating: {}'.format(name))
        self.name = name
        self.image_directory = image_dir
        self.image_prefix = ''
        self.COCO = COCO(anno_file)
        # self.debug_timer = Timer()
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1
            for i, v in enumerate(self.COCO.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k
            for k, v in self.json_category_id_to_contiguous_id.items()
        }
        # self._init_keypoints()
        self.keypoints = None

    def get_roidb(
        self,
        gt=False,
        proposal_file=None,
        min_proposal_size=2,
        proposal_limit=-1,
        crowd_filter_thresh=0
    ):
        """Return an roidb corresponding to the json dataset. Optionally:
           - include ground truth boxes in the roidb
           - add proposals specified in a proposals file
           - filter proposals based on a minimum side length
           - filter proposals that intersect with crowd regions
        """
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        for entry in roidb:
            self._prep_roidb_entry(entry)
        
        return roidb

    def _prep_roidb_entry(self, entry):
        """Adds empty metadata fields to an roidb entry."""
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        entry['image'] = os.path.join(
                self.image_directory, 
                self.image_prefix + entry['file_name'])

        for k in ['date_captured', 'url', 'license']:
            if k in entry:
                del entry[k]
