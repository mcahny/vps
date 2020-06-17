import os
import os.path as osp
from .coco import CocoDataset
from pycocotools.coco import COCO
import numpy as np
from .registry import DATASETS
from mmcv.parallel import DataContainer as DC
from .pipelines.formating import to_tensor
import pdb

@DATASETS.register_module
class CityscapesVideoOfsDataset(CocoDataset):
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
                 ref_ann_file=None,
                 offsets=None,
                 nframes_span_test=6):
        super(CityscapesVideoOfsDataset, self).__init__(        
                 ann_file=ann_file,
                 pipeline=pipeline,
                 data_root=data_root,
                 img_prefix=img_prefix,
                 seg_prefix=seg_prefix,
                 # flow_prefix=flow_prefix,
                 ref_prefix=ref_prefix,
                 proposal_file=proposal_file,
                 test_mode=test_mode,
                 ref_ann_file=ref_ann_file)

        if self.ref_ann_file is not None:
            self.ref_img_infos = self.load_ref_annotations(
                    self.ref_ann_file)
            self.iid2ref_img_infos = {x['id']:x for x in self.img_infos}
        self.offsets = offsets
        self.nframes_span_test = nframes_span_test

    CLASSES = ('person', 'rider', 'car', 'truck', 'bus', 
               'train', 'motorcycle', 'bicycle')

    # CLASSES_ALL = ('car','truck', 'person', 'rider','bicycle','bus',
    #                'motorcycle','train', 'road', 'sidewalk', 'building',
    #                'wall','fence','pole','traffic_light','traffic_sign',
    #                'vegetation','terrain','sky')
    # # all classes following the "cityscapes_panoptic.json"
    # trainId2cat = {0:'road',1:'sidewalk',2:'building',3:'wall',4:'fence',5:'pole',
    #             6:'traffic_light',7:'traffic_sign',8:'vegetation',9:'terrain',
    #             10:'sky',11:'person',12:'rider',13:'car',14:'truck',15:'bus',16:'train',
    #             17:'motorcycle',18:'bicycle'}

    # trainId2label = {11:1, 12:2, 13:3,  14:4, 15:5, 16:6, 17:7, 18:8, 
    #             0:9, 1:10, 2:11, 3:12, 4:13, 5:14, 6:15, 7:16, 8:17, 9:18, 10:19, 
    #             -1:0, 255:0}
    # label2semantic = {i+1:i+1 for i in trainId2cat.keys()}


    def load_ref_annotations(self, ann_file):
        self.ref_coco = COCO(ann_file)
        self.ref_cat_ids = self.ref_coco.getCatIds()
        self.ref_cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.ref_cat_ids)
        }
        self.ref_img_ids = self.ref_coco.getImgIds()
        img_infos = []
        for i in self.ref_img_ids:
            info = self.ref_coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    # def get_ref_ann_info(self, idx):
    #     img_id = self.ref_img_infos[idx]['id']
    #     ann_ids = self.ref_coco.getAnnIds(imgIds=[img_id])
    #     ann_info = self.ref_coco.loadAnns(ann_ids)
    #     return self._parse_ann_info(self.ref_img_infos[idx], ann_info)

    def get_ref_ann_info_by_iid(self, img_id, ref_img_info):
        # img_id = self.ref_img_infos[idx]['id']
        ann_ids = self.ref_coco.getAnnIds(imgIds=[img_id])
        ann_info = self.ref_coco.loadAnns(ann_ids)
        return self._parse_ann_info(ref_img_info, ann_info)


    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        # results['flow_prefix'] = self.flow_prefix
        results['ref_prefix'] = self.ref_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['ref_bbox_fields'] = []
        results['ref_mask_fields'] = []
        # results['trainId2label'] = self.trainId2label

    # def prepare_single_train_img(self, idx):
    #     img_info = self.img_infos[idx]
    #     ann_info = self.get_ann_info(idx)
    #     results = dict(img_info=img_info, ann_info=ann_info)
    #     if self.proposals is not None:
    #         results['proposals'] = self.proposals[idx]
    #     self.pre_pipeline(results)
    #     return self.pipeline(results)

    # def prepare_train_affine_img(self, idx):
    #     data = self.prepare_single_train_img(idx)
    #     ref_data = self.prepare_single_train_img(idx)
    #     if data is None or ref_data is None:
    #         return None

    #     data['ref_img'] = ref_data['img']
    #     data['ref_bboxes'] = ref_data['gt_bboxes']
    #     data['ref_obj_ids'] = ref_data['gt_obj_ids']
    #     data['ref_masks'] = ref_data['gt_masks']
    #     data['ref_labels'] = ref_data['gt_labels']
    #     # data['gt_pids'] BELOW
    #     # gt obj ids attribute did not exist in current annotation
    #     # we added it.
    #     ref_ids = ref_data['gt_obj_ids'].data.numpy().tolist()
    #     gt_ids = data['gt_obj_ids'].data.numpy().tolist()
    #     # compute matching of reference frame with current frame
    #     # 0 denote there is no matching
    #     # corresponding reference index ** +1 ** , if no match, give ZERO.
    #     gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]
    #     data['gt_pids'] = DC(to_tensor(gt_pids))
    #     return data


    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        iid = img_info['id']
        # offsets = [-1, 1] for Cityscapes_VPS
        offsets = self.offsets.copy()
        # random sampling of neighbor frame id [-1, 1] 
        while True:
            m = np.random.choice(offsets)
            if iid+m in self.ref_img_ids:
                break
            offsets.remove(m)
            # If all offset values fail, return None.
            if len(offsets)==0:
                return None
        # Reference image: information, annotations
        ref_iid = iid + m
        ref_img_info = self.iid2ref_img_infos[ref_iid]
        ref_ann_info = self.get_ref_ann_info_by_iid(
                ref_iid, ref_img_info)
        img_info['ref_id'] = ref_img_info['id']
        img_info['ref_filename'] = ref_img_info['filename']
        results = dict(img_info=img_info, ann_info=ann_info, 
                       ref_ann_info=ref_ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        ### semantic segmentation label (for target frame)
        # Cityscapes - specific filename
        seg_filename = osp.join(
                results['seg_prefix'], 
                results['ann_info']['seg_map'].replace(
                        'leftImg8bit', 'gtFine_color')).replace(
                                'newImg8bit', 'final_mask')
        results['ann_info']['seg_filename'] = seg_filename
        ### semantic segmentation label (for reference frame)
        # ===> Not being used in current training implementation.
        ref_seg_filename = osp.join(results['seg_prefix'], 
                results['ref_ann_info']['seg_map'].replace(
                        'leftImg8bit', 'gtFine_color')).replace(
                                'newImg8bit', 'final_mask')
        results['ref_ann_info']['seg_filename'] = ref_seg_filename

        data = self.pipeline(results)
        if data is None:
            return None
        ### tracking label
        ref_ids = data['ref_obj_ids'].data.numpy().tolist()
        gt_ids = data['gt_obj_ids'].data.numpy().tolist()
        gt_pids = [ref_ids.index(i)+1 if i in ref_ids else 0 for i in gt_ids]
        data['gt_pids'] = DC(to_tensor(gt_pids))
        return data



    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]

        prev_img_info = self.img_infos[idx-1] if idx%(self.nframes_span_test) > 0 else img_info
        img_info['ref_id'] = prev_img_info['id']-1
        img_info['ref_filename'] = prev_img_info['file_name']

        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_obj_ids = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])
                gt_obj_ids.append(ann['inst_id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_obj_ids = np.array(gt_obj_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_obj_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            obj_ids=gt_obj_ids,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann