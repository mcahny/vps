# ------------------------------------------------------------------
# Modified from:
#  Unified Panoptic Segmentation Network 
# (https://github.com/uber-research/UPSNet)
# ------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import pickle
import numpy as np
import os
import scipy.sparse

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO
from tools.config.config import config

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
