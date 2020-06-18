import os
import os.path as osp
from .coco import CocoDataset
import numpy as np
from .registry import DATASETS
from mmcv.parallel import DataContainer as DC
from .pipelines.formating import to_tensor
import pdb

@DATASETS.register_module
class CityscapesDataset(CocoDataset):
    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 seg_prefix=None,
                 # flow_prefix=None,
                 ref_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 offsets=None):
        super(CityscapesDataset, self).__init__(        
                 ann_file=ann_file,
                 pipeline=pipeline,
                 data_root=data_root,
                 img_prefix=img_prefix,
                 seg_prefix=seg_prefix,
                 # flow_prefix=flow_prefix,
                 ref_prefix=ref_prefix,
                 proposal_file=proposal_file,
                 test_mode=test_mode)

        self.offsets = offsets

    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 'train', 
               'motorcycle', 'bicycle')



    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['ref_prefix'] = self.ref_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []


    # to sample neighbor frame from offsets=[-1, +1]
    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        iid = img_info['id']
        filename = img_info['filename']
        # Cityscapes - specific filename
        fid = int(filename.split('_')[-2])
        suffix = '_'+filename.split('_')[-1]
        # offsets = [-1, 1] for Cityscapes
        offsets = self.offsets.copy()
        # random sampling of neighbor frame id [-1, 1] 
        m = np.random.choice(offsets)
        ref_filename = filename.replace(
                '%06d'%fid+suffix, 
                '%06d'%(fid+m)+suffix) if fid >= 1 else filename
        img_info['ref_filename'] = ref_filename
        results = dict(img_info=img_info, ann_info=ann_info)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        ### semantic segmentation label (only for target frame)
        # Cityscapes - specific filename
        seg_filename = osp.join(
                results['seg_prefix'], 
                results['ann_info']['seg_map'].replace(
                        'leftImg8bit',
                        'gtFine_labelTrainIds'))
        results['ann_info']['seg_filename'] = seg_filename
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        filename = img_info['filename']
        # Cityscapes - specific filename
        fid = int(filename.split('_')[-2])
        suffix = '_'+filename.split('_')[-1]
        # reference frame = past (-1) frame
        ref_filename = filename.replace(
                '%06d'%fid+suffix, 
                '%06d'%(fid-1)+suffix) if fid >= 1 else filename
        img_info['ref_filename'] = ref_filename
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

