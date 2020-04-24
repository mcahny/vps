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
# ---------------------------------------------------------------------------

from __future__ import print_function

import os
import sys
import torch
import torch.utils.data

import pickle, gzip
import numpy as np
import scipy.io as sio
import cv2
import json
import torch.multiprocessing as multiprocessing
import time
from PIL import Image, ImageDraw
from collections import defaultdict, Sequence
from collections import defaultdict, Sequence
from pycocotools.cocoeval import COCOeval

from upsnet.config.config import config
from upsnet.dataset.json_dataset import JsonDataset, extend_with_flipped_entries, filter_for_training, add_bbox_regression_targets
from upsnet.dataset.base_dataset import BaseDataset
from upsnet.rpn.assign_anchor import add_rpn_blobs
from upsnet.bbox import bbox_transform
from upsnet.bbox.sample_rois import sample_rois

import networkx as nx
from lib.utils.logging import logger
import pycocotools.mask as mask_util
import pdb
from .base_dataset import *
# panoptic visualization
vis_panoptic = False

class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue

            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                # per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0, 'iou': 0.0, 'tp':0, 'fp':0, 'fn':0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class, 'iou': iou, 'tp':tp, 'fp':fp, 'fn':fn}
            pq += pq_class
            sq += sq_class
            rq += rq_class
            print('isthing:',isthing, 'cat:',label_info['name'], 'pq:', pq_class)
        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results

class CityscapesExt(BaseDataset):

    def __init__(self, image_sets, flip=False, proposal_files=None, phase='train', result_path=''):

        super(CityscapesExt, self).__init__()
        #### Re-configuration for Viper dataset
        config.dataset.dataset_path = '../../data/cityscapes_ext/'
        config.dataset.name = 'demo_munster'

        self.image_dirs = {
            'train': os.path.join(config.dataset.dataset_path, 'train/img'),
            'val': os.path.join(config.dataset.dataset_path, 'demo_munster/img'),
            'test': os.path.join(config.dataset.dataset_path, 'test/img'),
        }

        self.anno_files = {
            'train': 'instances_train_01_city_coco_rle.json',
            'val': 'instances_val_01_city_im_munster.json',
            'test': 'instances_val_01_city_coco_rle.json',
        }
        self.panoptic_json_file = '/data2/video_panoptic/data/cityscapes_ext/cityscapes_ext_panoptic_val_video.json'
        self.panoptic_gt_folder = '/data2/video_panoptic/data/cityscapes_ext/val/panoptic_video/'

        self.flip = flip
        self.result_path = result_path
        self.num_classes = 9
        self.phase = phase
        self.image_sets = image_sets

        if image_sets[0] == 'demoVideo':
            assert len(image_sets) == 1
            assert phase == 'test'
            im_path = [_.strip() for _ in open('data/cityscapes/split/demoVideo_img.txt', 'r').readlines()]
            self.roidb = [{'image': _, 'flipped': False} for _ in im_path]
            return
        if proposal_files is None:
            proposal_files = [None] * len(image_sets)

        if phase == 'train' and len(image_sets) > 1:
            # combine multiple datasets
            roidbs = []
            for image_set, proposal_file in zip(image_sets, proposal_files):
                dataset = JsonDataset('viper_' + image_set,
                                      image_dir=self.image_dirs[image_set],
                                      anno_file=os.path.join(config.dataset.dataset_path, 'annotations', self.anno_files[image_set]))
                roidb = dataset.get_roidb(gt=True, proposal_file=proposal_file, crowd_filter_thresh=config.train.crowd_filter_thresh)
                if flip:
                    if logger:
                        logger.info('Appending horizontally-flipped training examples...')
                    extend_with_flipped_entries(roidb, dataset)
                roidbs.append(roidb)
            roidb = roidbs[0]
            for r in roidbs[1:]:
                roidb.extend(r)
            roidb = filter_for_training(roidb)
            add_bbox_regression_targets(roidb)

        else:
            assert len(image_sets) == 1
            self.dataset = JsonDataset('cityscapes_ext_' + image_sets[0],
                                       image_dir=self.image_dirs[image_sets[0]],
                                       anno_file=os.path.join(config.dataset.dataset_path,
                                       self.anno_files[image_sets[0]]))
            roidb = self.dataset.get_roidb(gt=True, proposal_file=proposal_files[0],
                                           crowd_filter_thresh=config.train.crowd_filter_thresh if phase != 'test' else 0)
            if flip:
                if logger:
                    logger.info('Appending horizontally-flipped training examples...')
                extend_with_flipped_entries(roidb, self.dataset)
            if phase != 'test':
                roidb = filter_for_training(roidb)
                add_bbox_regression_targets(roidb)

        self.roidb = roidb

    def __getitem__(self, index):
        blob = defaultdict(list)
        im_blob, im_scales = self.get_image_blob([self.roidb[index]])
        if config.network.has_rpn:
            if self.phase != 'test':
                add_rpn_blobs(blob, im_scales, [self.roidb[index]])
                data = {'data': im_blob,
                        'im_info': blob['im_info']}
                label = {'roidb': blob['roidb'][0]}
                for stride in config.network.rpn_feat_stride:
                    label.update({
                        'rpn_labels_fpn{}'.format(stride): blob['rpn_labels_int32_wide_fpn{}'.format(stride)].astype(
                            np.int64),
                        'rpn_bbox_targets_fpn{}'.format(stride): blob['rpn_bbox_targets_wide_fpn{}'.format(stride)],
                        'rpn_bbox_inside_weights_fpn{}'.format(stride): blob[
                            'rpn_bbox_inside_weights_wide_fpn{}'.format(stride)],
                        'rpn_bbox_outside_weights_fpn{}'.format(stride): blob[
                            'rpn_bbox_outside_weights_wide_fpn{}'.format(stride)]
                    })
            else:
                data = {'data': im_blob,
                        'im_info': np.array([[im_blob.shape[-2],
                                              im_blob.shape[-1],
                                             im_scales[0]]], np.float32),
                        }
                label = {'roidb': self.roidb[index]}
        else:
            pass

        if config.network.has_fcn_head:
            if self.phase != 'test':
                seg_gt = np.array(Image.open(self.roidb[index]['image'].replace('images', 'labels').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png')))
                if self.roidb[index]['flipped']:
                    seg_gt = np.fliplr(seg_gt)
                seg_gt = cv2.resize(seg_gt, None, None, fx=im_scales[0], fy=im_scales[0], interpolation=cv2.INTER_NEAREST)
                label.update({'seg_gt': seg_gt})
                label.update({'gt_classes': label['roidb']['gt_classes']})
                label.update({'mask_gt': np.zeros((len(label['gt_classes']), im_blob.shape[-2], im_blob.shape[-1]))})
                for i in range(len(label['gt_classes'])):
                    img = Image.new('L', (int(im_blob.shape[-1] / im_scales[0]), int(im_blob.shape[-2] / im_scales[0])), 0)
                    for j in range(len(label['roidb']['segms'][i])):
                        ImageDraw.Draw(img).polygon(tuple(label['roidb']['segms'][i][j]), outline=1, fill=1)
                    label['mask_gt'][i] = cv2.resize(np.array(img), None, None, fx=im_scales[0], fy=im_scales[0], interpolation=cv2.INTER_NEAREST)
                if config.train.fcn_with_roi_loss:
                    gt_boxes = label['roidb']['boxes'][np.where(label['roidb']['gt_classes'] > 0)[0]]
                    gt_boxes = np.around(gt_boxes * im_scales[0]).astype(np.int32)
                    label.update({'seg_roi_gt': np.zeros((len(gt_boxes), config.network.mask_size, config.network.mask_size), dtype=np.int64)})
                    for i in range(len(gt_boxes)):
                        if gt_boxes[i][3] == gt_boxes[i][1]:
                            gt_boxes[i][3] += 1
                        if gt_boxes[i][2] == gt_boxes[i][0]:
                            gt_boxes[i][2] += 1
                        label['seg_roi_gt'][i] = cv2.resize(seg_gt[gt_boxes[i][1]:gt_boxes[i][3], gt_boxes[i][0]:gt_boxes[i][2]], (config.network.mask_size, config.network.mask_size), interpolation=cv2.INTER_NEAREST)
            else:
                pass

        return data, label, index

    def get_image_blob(self, roidb):
        """Builds an input blob from the images in the roidb at the specified
        scales.
        """
        num_images = len(roidb)
        # Sample random scales to use for each image in this batch
        if self.phase == 'train':
            scale_inds = np.random.randint(
                0, high=len(config.train.scales), size=num_images
            )
        else:
            scale_inds = np.random.randint(
                0, high=len(config.test.scales), size=num_images
            )
        processed_ims = []
        im_scales = []
        for i in range(num_images):
            im = cv2.imread(roidb[i]['image'])
            assert im is not None, \
                'Failed to read image \'{}\''.format(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            if self.phase == 'train':
                target_size = config.train.scales[scale_inds[i]]
                im, im_scale = self.prep_im_for_blob(
                    im, config.network.pixel_means, [target_size], config.train.max_size
                )
            else:
                target_size = config.test.scales[scale_inds[i]]
                im, im_scale = self.prep_im_for_blob(
                    im, config.network.pixel_means, [target_size], config.test.max_size
                )
            im_scales.append(im_scale[0])
            processed_ims.append(im[0].transpose(2, 0, 1))

        # Create a blob to hold the input images
        assert len(processed_ims) == 1
        blob = processed_ims[0]

        return blob, im_scales


    def get_pallete(self):

        pallete_raw = np.zeros((256, 3)).astype('uint8')
        pallete = np.zeros((256, 3)).astype('uint8')

        pallete_raw[5, :] =  [111,  74,   0]
        pallete_raw[6, :] =  [ 81,   0,  81]
        pallete_raw[7, :] =  [128,  64, 128]
        pallete_raw[8, :] =  [244,  35, 232]
        pallete_raw[9, :] =  [250, 170, 160]
        pallete_raw[10, :] = [230, 150, 140]
        pallete_raw[11, :] = [ 70,  70,  70]
        pallete_raw[12, :] = [102, 102, 156]
        pallete_raw[13, :] = [190, 153, 153]
        pallete_raw[14, :] = [180, 165, 180]
        pallete_raw[15, :] = [150, 100, 100]
        pallete_raw[16, :] = [150, 120,  90]
        pallete_raw[17, :] = [153, 153, 153]
        pallete_raw[18, :] = [153, 153, 153]
        pallete_raw[19, :] = [250, 170,  30]
        pallete_raw[20, :] = [220, 220,   0]
        pallete_raw[21, :] = [107, 142,  35]
        pallete_raw[22, :] = [152, 251, 152]
        pallete_raw[23, :] = [ 70, 130, 180]
        pallete_raw[24, :] = [220,  20,  60]
        pallete_raw[25, :] = [255,   0,   0]
        pallete_raw[26, :] = [  0,   0, 142]
        pallete_raw[27, :] = [  0,   0,  70]
        pallete_raw[28, :] = [  0,  60, 100]
        pallete_raw[29, :] = [  0,   0,  90]
        pallete_raw[30, :] = [  0,   0, 110]
        pallete_raw[31, :] = [  0,  80, 100]
        pallete_raw[32, :] = [  0,   0, 230]
        pallete_raw[33, :] = [119,  11,  32]

        train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]

        for i in range(len(train2regular)):
            pallete[i, :] = pallete_raw[train2regular[i], :]

        pallete = pallete.reshape(-1)

        # return pallete_raw
        return pallete

    def evaluate_ssegs(self, pred_segmentations, res_file_folder, pred_segm_names):
        self.write_segmentation_result(pred_segmentations, res_file_folder,
                                       pred_segm_names)

        # viper_class_mapping = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7,
        #         8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:15,16:16,17:17,
        #         18:18,19:19,20:20,21:21,22:22,-1:255,255:255}

        confusion_matrix = np.zeros((config.dataset.num_seg_classes, config.dataset.num_seg_classes))

        for i, roidb in enumerate(self.roidb):
            
            img_name = roidb['image']
            iid = img_name.split('/')[-1].replace('newImg8bit','final_mask').replace('leftImg8bit','gtFine_color')
            seg_name = os.path.join(config.dataset.dataset_path, roidb['mode'],
                'labelmap',iid)
            seg_gt = np.array(Image.open(seg_name)).astype('float32')

            resid = iid.replace('_final_mask','').replace('_gtFine_color','')
            res_save_path = os.path.join(res_file_folder, resid)
            seg_pred = Image.open(res_save_path)
            seg_pred = np.array(seg_pred.resize((seg_gt.shape[1], seg_gt.shape[0]), Image.NEAREST))

            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

            confusion_matrix += self.get_confusion_matrix(seg_gt, seg_pred, config.dataset.num_seg_classes)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        evaluation_results = {'meanIU':mean_IU, 'IU_array':IU_array, 'confusion_matrix': confusion_matrix}

        def convert_confusion_matrix(confusion_matrix):
            cls_sum = confusion_matrix.sum(axis=1)
            confusion_matrix = confusion_matrix / cls_sum.reshape((-1, 1))
            return confusion_matrix

        # logger.info('evaluate segmentation:')
        print('evaluate segmentation:')
        meanIU = evaluation_results['meanIU']
        IU_array = evaluation_results['IU_array']
        confusion_matrix = convert_confusion_matrix(evaluation_results['confusion_matrix'])
        print('IU_array:')
        for i in range(len(IU_array)):
            print('%.5f' % IU_array[i])
        print('meanIU:%.5f' % meanIU)
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        import re
        confusion_matrix = re.sub('[\[\]]', '', np.array2string(confusion_matrix, separator='\t'))

        print('confusion_matrix:')
        print(confusion_matrix)

    def write_segmentation_result(self, segmentation_results, res_file_folder, 
                                  pred_segm_names):
        """
        Write the segmentation result to result_file_folder
        :param segmentation_results: the prediction result
        :param result_file_folder: the saving folder
        :return: [None]
        """
        if not os.path.exists(res_file_folder):
            os.mkdir(res_file_folder)

        pallete = self.get_pallete()
        # for i, roidb in enumerate(self.roidb):

        #     seg_pathes = os.path.split(roidb['image'])
        #     res_image_name = seg_pathes[-1][:-len('_leftImg8bit.png')]
        #     res_save_path = os.path.join(res_file_folder, res_image_name + '.png')
        for i, pred_segm_name in enumerate(pred_segm_names):
            if 'leftImg8bit.png' in pred_segm_name or 'newImg8bit.png' in pred_segm_name:
                res_save_path = os.path.join(res_file_folder, pred_segm_name.replace('_leftImg8bit.png','.png').replace('_newImg8bit.png','.png'))
            elif '.jpg' in pred_segm_name:
                res_save_path = os.path.join(res_file_folder, pred_segm_name.replace('.jpg', '.png'))

            segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))
            # pdb.set_trace()
            segmentation_result = Image.fromarray(segmentation_result)
            segmentation_result.putpalette(pallete)
            segmentation_result.save(res_save_path)



    def evaluate_panoptic(self, pred_pans_2ch, output_dir, is_track=False, n_video=0, save_name=None):
        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))

        from panopticapi.utils import IdGenerator

        def get_gt(pan_gt_json_file=None, pan_gt_folder=None):
            if pan_gt_json_file is None:
                pan_gt_json_file = self.panoptic_json_file
            if pan_gt_folder is None:
                pan_gt_folder = self.panoptic_gt_folder
            with open(pan_gt_json_file, 'r') as f:
                pan_gt_json = json.load(f)
            # pdb.set_trace()
            # files = [item['file_name'] for item in pan_gt_json['images']]
            files = [item['file_name'].replace('leftImg8bit','gtFine_color').replace('newImg8bit','final_mask') for item in pan_gt_json['images']]
            # if 'viper' in pan_gt_folder:
            #     files = [_.split('/')[-1].replace('.jpg', '.png') for _ in files]
            cpu_num = multiprocessing.cpu_count()
            files_split = np.array_split(files, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            for proc_id, files_set in enumerate(files_split):
                p = workers.apply_async(BaseDataset._load_image_single_core, (proc_id, files_set, pan_gt_folder))
                processes.append(p)
            workers.close()
            workers.join()
            pan_gt_all = []
            for p in processes:
                pan_gt_all.extend(p.get())

            categories = pan_gt_json['categories']
            categories = {el['id']: el for el in categories}
            color_gererator = IdGenerator(categories)
            return pan_gt_all, pan_gt_json, categories, color_gererator

        def get_pred(pan_2ch_all, color_gererator, cpu_num=0):
            if cpu_num is 0:
                cpu_num = multiprocessing.cpu_count()
            pan_2ch_split = np.array_split(pan_2ch_all, cpu_num)
            # pan_2ch_split = np.array_split(pan_2ch_all, 1)
            workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
                #### DEBUG
                # pdb.set_trace()
                # p = self.converter_2ch_track_core(proc_id, pan_2ch_set, color_gererator)
                if is_track:
                    p = workers.apply_async(self.converter_2ch_track_core, (proc_id, pan_2ch_set, color_gererator))
                else:
                    p = workers.apply_async(self.converter_2ch_single_core, (proc_id, pan_2ch_set, color_gererator))
                processes.append(p)
            workers.close()
            workers.join()
            annotations, pan_all = [], []
            for p in processes:
                p = p.get()
                annotations.extend(p[0])
                pan_all.extend(p[1])
            pan_json = {'annotations': annotations}
            return pan_all, pan_json

        def get_pred_large(pan_2ch_all, color_gererator, vid_num=100):
            assert vid_num > multiprocessing.cpu_count()
            nparts = 5
            pan_2ch_parts = np.array_split(pan_2ch_all, nparts) # 20 videos each part
            cpu_num = vid_num // nparts # 20
            annotations, pan_all = [],[]
            for idx, pan_2ch_part in enumerate(pan_2ch_parts):
                pan_2ch_split = np.array_split(pan_2ch_part, cpu_num)
                workers = multiprocessing.Pool(processes=cpu_num)
                processes = []
                for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
                    print('part %d:'%(idx), 'proc_id:',proc_id)
                    if is_track:
                        p = workers.apply_async(self.converter_2ch_track_core, (proc_id, pan_2ch_set, color_gererator))
                    else:
                        p = workers.apply_async(self.converter_2ch_single_core, (proc_id, pan_2ch_set, color_gererator))
                    processes.append(p)
                workers.close()
                workers.join()
                
                for p in processes:
                    p = p.get()
                    annotations.extend(p[0])
                    pan_all.extend(p[1])

            pan_json = {'annotations': annotations}
            return pan_all, pan_json



        def save_image(images, save_folder, gt_json, colors=None):
            os.makedirs(save_folder, exist_ok=True)
            names = [os.path.join(save_folder, item['file_name'].replace('_leftImg8bit', '').replace('_newImg8bit','').replace('jpg', 'png').replace('jpeg', 'png')) for item in gt_json['images']]
            cpu_num = multiprocessing.cpu_count()
            images_split = np.array_split(images, cpu_num)
            names_split = np.array_split(names, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            for proc_id, (images_set, names_set) in enumerate(zip(images_split, names_split)):
                workers.apply_async(BaseDataset._save_image_single_core, (proc_id, images_set, names_set, colors))
            workers.close()
            workers.join()

        def vpq_compute(gt_jsons, pred_jsons, gt_pans, pred_pans, categories, vid_num, nframes=2, save_name=None):
            start_time = time.time()
            # from json and from numpy
            gt_image_jsons = gt_jsons['images']
            gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
            cpu_num = multiprocessing.cpu_count()
            # assert cpu_num >= vid_num, "vid_num must be smaller than cpu_num."
            if cpu_num >= vid_num:
                pq_stat = PQStat()
                gt_jsons_split, pred_jsons_split = np.array_split(gt_jsons, vid_num), np.array_split(pred_jsons, vid_num)
                gt_pans_split, pred_pans_split = np.array_split(gt_pans, vid_num), np.array_split(pred_pans, vid_num)
                gt_image_jsons_split = np.array_split(gt_image_jsons, vid_num)

                workers = multiprocessing.Pool(processes=vid_num)
                processes = []
                for proc_id, (gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set) in enumerate(zip(gt_jsons_split, pred_jsons_split, gt_pans_split, pred_pans_split, gt_image_jsons_split)):
                    ####****************** DEBUG ******************
                    # vpq = Viper._vpq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories, nframes=nframes)
                    p = workers.apply_async(CityscapesExt._vpq_compute_single_core, (proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories, nframes))
                    processes.append(p)
                workers.close()
                workers.join()
                for p in processes:
                    pq_stat += p.get()
            else:
                nparts = 5
                cpu_num = vid_num // nparts
                print('vid_num:',vid_num, 'nparts:',nparts,'cpu_num:', cpu_num)
                pq_stat = PQStat()
                gt_jsons_parts, pred_jsons_parts = np.array_split(gt_jsons,nparts), np.array_split(pred_jsons, nparts)
                gt_pans_parts, pred_pans_parts = np.array_split(gt_pans, nparts), np.array_split(pred_pans, nparts)
                gt_image_jsons_parts = np.array_split(gt_image_jsons, nparts)

                for idx, (gt_jsons_part, pred_jsons_part, gt_pans_part, pred_pans_part, gt_image_jsons_part) in enumerate(zip(gt_jsons_parts, pred_jsons_parts, gt_pans_parts, pred_pans_parts, gt_image_jsons_parts)):
                    # devide by cpu_num
                    gt_jsons_split, pred_jsons_split = np.array_split(gt_jsons_part, cpu_num), np.array_split(pred_jsons_part, cpu_num)
                    gt_pans_split, pred_pans_split = np.array_split(gt_pans_part, cpu_num), np.array_split(pred_pans_part, cpu_num)
                    gt_image_jsons_split = np.array_split(gt_image_jsons_part, cpu_num)

                    workers = multiprocessing.Pool(processes=cpu_num)
                    processes = []
                    for proc_id, (gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set) in enumerate(zip(gt_jsons_split, pred_jsons_split, gt_pans_split, pred_pans_split, gt_image_jsons_split)):
                        print('part %d:'%(idx), 'proc_id:',proc_id)
                        # p = CityscapesExt._vpq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories, nframes)
                        # pdb.set_trace()
                        # p.get()
                        p = workers.apply_async(CityscapesExt._vpq_compute_single_core, (proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories, nframes))
                        processes.append(p)
                    workers.close()
                    workers.join()
                    for p in processes:
                        pq_stat += p.get()


            metrics = [("All", None), ("Things", True), ("Stuff", False)]
            results = {}
            for name, isthing in metrics:
                results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
                if name == 'All':
                    results['per_class'] = per_class_results

            print("============== for %d-frames ============="%(nframes))
            print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
            print("-" * (10 + 7 * 4))
            for name, _isthing in metrics:
                print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(name, 100 * results[name]['pq'], 100 * results[name]['sq'], 100 * results[name]['rq'], results[name]['n']))

            print("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}".format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN"))
            for idx, result in results['per_class'].items():
                print("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}".format(idx, 100 * result['pq'], 100 * result['sq'], 100 * result['rq'], result['iou'], result['tp'], result['fp'], result['fn']))

            if save_name is not None:
                with open(save_name ,'w') as f:
                    print("============== for %d-frames ============="%(nframes), file=f)
                    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"), file=f)
                    print("-" * (10 + 7 * 4))
                    for name, _isthing in metrics:
                        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(name, 100 * results[name]['pq'], 100 * results[name]['sq'], 100 * results[name]['rq'], results[name]['n']), file=f)

                    print("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}".format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN"), file=f)
                    for idx, result in results['per_class'].items():
                        print("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}".format(idx, 100 * result['pq'], 100 * result['sq'], 100 * result['rq'], result['iou'], result['tp'], result['fp'], result['fn']), file=f)
            
            t_delta = time.time() - start_time
            print("Time elapsed: {:0.2f} seconds".format(t_delta))

            return results
                    
                
        def pq_compute(gt_jsons, pred_jsons, gt_pans, pred_pans, categories, save_name=None):
            start_time = time.time()
            # from json and from numpy
            gt_image_jsons = gt_jsons['images']
            gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
            cpu_num = multiprocessing.cpu_count()
            gt_jsons_split, pred_jsons_split = np.array_split(gt_jsons, cpu_num), np.array_split(pred_jsons, cpu_num)
            gt_pans_split, pred_pans_split = np.array_split(gt_pans, cpu_num), np.array_split(pred_pans, cpu_num)
            gt_image_jsons_split = np.array_split(gt_image_jsons, cpu_num)

            workers = multiprocessing.Pool(processes=cpu_num)
            processes = []
            for proc_id, (gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set) in enumerate(zip(gt_jsons_split, pred_jsons_split, gt_pans_split, pred_pans_split, gt_image_jsons_split)):
                # p = workers.apply_async(self.pq_compute_single_core, (proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories))
                # dbg = self.pq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories)
                # pdb.set_trace()
                # dbg = CityscapesExt.pq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories)
                # pdb.set_trace()
                # dbg.pq_average(categories, isthing=True)
                p = workers.apply_async(CityscapesExt._pq_compute_single_core, (proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories))
                processes.append(p)
            workers.close()
            workers.join()
            pq_stat = PQStat()
            for p in processes:
                pq_stat += p.get()
            metrics = [("All", None), ("Things", True), ("Stuff", False)]
            results = {}
            for name, isthing in metrics:

                results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
                if name == 'All':
                    results['per_class'] = per_class_results

            print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
            print("-" * (10 + 7 * 4))
            for name, _isthing in metrics:
                print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(name, 100 * results[name]['pq'], 100 * results[name]['sq'], 100 * results[name]['rq'], results[name]['n']))

            print("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}".format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN"))
            for idx, result in results['per_class'].items():
                print("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}".format(idx, 100 * result['pq'], 100 * result['sq'], 100 * result['rq'], result['iou'], result['tp'], result['fp'], result['fn']))

            if save_name is not None:
                with open(save_name ,'w') as f:
                    print("============== for 1-frame =============", file=f)
                    print("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"), file=f)
                    print("-" * (10 + 7 * 4))
                    for name, _isthing in metrics:

                        print("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(name, 100 * results[name]['pq'], 100 * results[name]['sq'], 100 * results[name]['rq'], results[name]['n']), file=f)

                    print("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}".format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN"), file=f)
                    for idx, result in results['per_class'].items():
                        print("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}".format(idx, 100 * result['pq'], 100 * result['sq'], 100 * result['rq'], result['iou'], result['tp'], result['fp'], result['fn']), file=f)

            t_delta = time.time() - start_time
            print("Time elapsed: {:0.2f} seconds".format(t_delta))
            return results

        #### if eval for test-dev, since there is no gt we simply retrieve image names from image_info json files
        with open(self.panoptic_json_file, 'r') as f:
            pano_json = json.load(f)
            categories = pano_json['categories']
            categories = {el['id']: el for el in categories}
            color_gererator = IdGenerator(categories)

        tar_json_file = self.panoptic_json_file.replace(self.panoptic_json_file.split('/')[-1], self.anno_files['val'])
        with open(os.path.join(tar_json_file)) as f:
            gt_json = json.load(f)
            gt_json['images'] = sorted(gt_json['images'], key=lambda x: x['id'])
            # pdb.set_trace()
            
        #### other wise:
        # gt_pans, gt_json, categories, color_gererator = get_gt()
        
        if n_video <=  32:
            pred_pans, pred_json = get_pred(pred_pans_2ch, color_gererator, 
                                        cpu_num=n_video)
        else:
            pred_pans, pred_json = get_pred_large(pred_pans_2ch, color_gererator,
                                        vid_num=n_video)

        save_image(pred_pans_2ch, os.path.join(output_dir, 'pan_2ch'), gt_json)
        save_image(pred_pans, os.path.join(output_dir, 'pan'), gt_json)
        pdb.set_trace()
        json.dump(gt_json, open(os.path.join(output_dir, 'gt.json'), 'w'))
        json.dump(pred_json, open(os.path.join(output_dir, 'pred.json'), 'w'))

        results = pq_compute(gt_json, pred_json, gt_pans, pred_pans, categories, save_name=save_name+'_vpq_nf01.txt')

        # for nframes in [2,3,4,5,10,15]:
        for nframes in [2,3,4,5]:
            results = vpq_compute(gt_json, pred_json, gt_pans, pred_pans, categories, vid_num=n_video, nframes=nframes, save_name=save_name+'_vpq_nf%02d.txt'%(nframes))
        # results = pq_compute(gt_json, pred_json, gt_pans, pred_pans, categories)

        return results

    @staticmethod
    def _vpq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories, nframes=2):
        OFFSET = 256 * 256 * 256
        VOID = 0
        SIZE_THR = 4**2
        vpq_stat = PQStat()
        for idx in range(0, len(pred_pans_set)-nframes+1): # nframes:2 ==> i: 0~58
            start_idx = time.time()
            vid_pan_gt = []
            vid_pan_pred = []
            gt_segms_list = []
            pred_segms_list = []
            sub_ann_set = \
                zip(gt_jsons_set[idx:idx+nframes], 
                    pred_jsons_set[idx:idx+nframes],
                    gt_pans_set[idx:idx+nframes],
                    pred_pans_set[idx:idx+nframes],
                    gt_image_jsons_set[idx:idx+nframes])
            #### Output VPQ value for "nframe" volume.
            # Step1. to merge jsons and pan maps in the volume.

            for gt_json, pred_json, gt_pan, pred_pan, gt_image_json in sub_ann_set:
                
                gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
                pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
                pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256

                gt_segms = {el['id']: el for el in gt_json['segments_info']}
                pred_segms = {el['id']: el for el in pred_json['segments_info']}
                # predicted segments area calculation + prediction sanity checks
                pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
                labels, labels_cnt = np.unique(pan_pred, return_counts=True)
                for label, label_cnt in zip(labels, labels_cnt):
                    if label not in pred_segms:
                        if label == VOID:
                            continue
                        raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
                    pred_segms[label]['area'] = label_cnt
                    pred_labels_set.remove(label)
                    if pred_segms[label]['category_id'] not in categories:
                        raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
                if len(pred_labels_set) != 0:
                    raise KeyError(
                        'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))
                #### Collect frame-lavel pan_map, jsons, etc.
                vid_pan_gt.append(pan_gt)
                vid_pan_pred.append(pan_pred)
                gt_segms_list.append(gt_segms)
                pred_segms_list.append(pred_segms)
            # Step 2. stack the collected elements ==> this is a unit
            vid_pan_gt = np.stack(vid_pan_gt) # [nf,1080,1920]
            vid_pan_pred = np.stack(vid_pan_pred) # [nf,1080,1920]
            vid_gt_segms, vid_pred_segms = {}, {}
            for gt_segms, pred_segms in zip(gt_segms_list, pred_segms_list):
                # merge 'area' only for gt_segms
                for k in gt_segms.keys():
                    if not k in vid_gt_segms:
                        vid_gt_segms[k] = gt_segms[k]
                    else:
                        vid_gt_segms[k]['area'] += gt_segms[k]['area']
                # merge 'area' only for pred_segms
                for k in pred_segms.keys():
                    if not k in vid_pred_segms:
                        vid_pred_segms[k] = pred_segms[k]
                    else:
                        vid_pred_segms[k]['area'] += pred_segms[k]['area']
            # Step3. Confusion matrix calculation
            vid_pan_gt_pred = vid_pan_gt.astype(np.uint64) * OFFSET + vid_pan_pred.astype(np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(vid_pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_small = set()
            gt_matched = set()
            pred_matched = set()
            tp = 0
            fp = 0
            fn = 0

            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                pred_area = (vid_pan_pred == pred_label).sum()
                gt_area = (vid_pan_gt == gt_label).sum()
                #### SKIP SMALL OBJECTS FIRST
                if gt_area < SIZE_THR:
                    gt_small.add(gt_label)
                    continue
                if gt_label not in vid_gt_segms:
                    continue
                if pred_label not in vid_pred_segms:
                    continue
                if vid_gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if vid_gt_segms[gt_label]['category_id'] != \
                        vid_pred_segms[pred_label]['category_id']:
                    continue

                union = pred_area + gt_area - intersection - gt_pred_map.get((VOID, pred_label),0)
                iou = intersection / union
                # ignore invalid iou value
                assert iou <= 1.0, 'INVALID IOU VALUE : %d'%(gt_label)
                if iou > 0.5:
                    vpq_stat[vid_gt_segms[gt_label]['category_id']].tp += 1
                    vpq_stat[vid_gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    tp += 1

            # count false positives
            crowd_labels_dict = {}
            for gt_label, gt_info in vid_gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                if gt_label in gt_small:
                    continue
                vpq_stat[gt_info['category_id']].fn += 1
                fn += 1

            # count false positives
            for pred_label, pred_info in vid_pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                vpq_stat[pred_info['category_id']].fp += 1
                fp += 1
            # print('Core %d ==> frame %d, took: %4f sec.'%(proc_id, idx, time.time()-start_idx))
        print('Core: {}, all {} images processed'.format(proc_id, len(pred_pans_set)))
        return vpq_stat


    @staticmethod
    def _pq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories):
        OFFSET = 256 * 256 * 256
        VOID = 0
        # SIZE_THR = 4**2
        SIZE_THR = 0
        pq_stat = PQStat()
        for idx, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(zip(gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set)):
            # if idx % 100 == 0:
            #     logger.info('Compute pq -> Core: {}, {} from {} images processed'.format(proc_id, idx, len(gt_jsons_set)))
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256
            gt_segms = {el['id']: el for el in gt_json['segments_info']}
            pred_segms = {el['id']: el for el in pred_json['segments_info']}

            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in categories:
                    raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))

            # confusion matrix calculation
            pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_small = set()
            gt_matched = set()
            pred_matched = set()
            tp = 0
            fp = 0
            fn = 0

            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                pred_area = (pan_pred == pred_label).sum()
                gt_area = (pan_gt == gt_label).sum()
                #### SKIP SMALL OBJECTS
                if gt_area < SIZE_THR:
                    gt_small.add(gt_label)
                    continue

                if gt_label not in gt_segms:
                    continue
                if pred_label not in pred_segms:
                    continue
                if gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                    continue
        
                # pred_small.add(pred_label)
                union = pred_area + gt_area - intersection - gt_pred_map.get((VOID, pred_label),0)
                iou = intersection / union
                # ignore invalid iou value
                if iou > 1.0:
                    print('INVALID IOU VALUE :',gt_label)
                    continue

                if iou > 0.5:
                    pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                    pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    tp += 1

            # count false positives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                if gt_label in gt_small:
                    continue
                pq_stat[gt_info['category_id']].fn += 1
                fn += 1

            # count false positives
            for pred_label, pred_info in pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with VOID
                intersection = gt_pred_map.get((VOID, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                pq_stat[pred_info['category_id']].fp += 1
                fp += 1
        # logger.info('Compute pq -> Core: {}, all {} images processed'.format(proc_id, len(gt_jsons_set)))
        return pq_stat


    def converter_2ch_single_core(self, proc_id, pan_2ch_set, color_gererator):
        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))
        from panopticapi.utils import rgb2id
        # viper_new2old = {13:1,14:2,15:3,16:4,17:5,18:6,19:7,20:8,21:9,22:10,
        #         0:11,1:12,2:13,3:14,4:15,5:16,6:17,7:18,8:19,9:20,10:21,11:22,12:23}
        # viper_new2old = {k:k for k in range(23)}
        #### Now it is corrected.
        # viper_new2old[14]=24
        OFFSET = 1000
        VOID = 255
        annotations, pan_all = [], []
        for idx in range(len(pan_2ch_set)):
            pan_2ch = np.uint32(pan_2ch_set[idx])
            pan = OFFSET * pan_2ch[:, :, 0] + pan_2ch[:, :, 1]
            pan_format = np.zeros((pan_2ch.shape[0], pan_2ch.shape[1], 3), dtype=np.uint8)
            l = np.unique(pan)
            segm_info = []
            for el in l:
                sem = el // OFFSET
                if sem == VOID:
                    continue
                mask = pan == el
                #### viper class mapping
                # sem = np.int64(viper_new2old[sem])
                # if vis_panoptic:
                #     color = color_gererator.categories[sem]['color']
                # else:
                #     color = color_gererator.get_color(sem)
                color = color_gererator.get_color(sem)
                pan_format[mask] = color
                index = np.where(mask)
                x = index[1].min()
                y = index[0].min()
                width = index[1].max() - x
                height = index[0].max() - y
                segm_info.append({"category_id": sem.item(), "iscrowd": 0, "id": int(rgb2id(color)), "bbox": [x.item(), y.item(), width.item(), height.item()], "area": mask.sum().item()})
            annotations.append({"segments_info": segm_info})

            if vis_panoptic:
                pan_format = Image.fromarray(pan_format)
                draw = ImageDraw.Draw(pan_format)
                for el in l:
                    sem = el // OFFSET
                    if sem == VOID:
                        continue
                    if color_gererator.categories[sem]['isthing'] and el % OFFSET != 0:
                        mask = ((pan == el) * 255).astype(np.uint8)
                        _, contour, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        for c in contour:
                            c = c.reshape(-1).tolist()
                            if len(c) < 4:
                                print('warning: invalid contour')
                                continue
                            draw.line(c, fill='white', width=2)
                pan_format = np.array(pan_format)
            pan_all.append(pan_format)
        return annotations, pan_all

    def converter_2ch_track_core(self, proc_id, pan_2ch_set, color_gererator):
        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))
        from panopticapi.utils import rgb2id
        # viper_new2old = {13:1,14:2,15:3,16:4,17:5,18:6,19:7,20:8,21:9,22:10,
        #         0:11,1:12,2:13,3:14,4:15,5:16,6:17,7:18,8:19,9:20,10:21,11:22,12:23}
        # viper_new2old = {k:k for k in range(23)}
        # viper_new2old[14]=24
        OFFSET = 1000
        VOID = 255
        annotations, pan_all = [], []
        # reference dict to used color
        inst2color = {}
        for idx in range(len(pan_2ch_set)):
            pan_2ch = np.uint32(pan_2ch_set[idx])
            # pan = OFFSET * pan_2ch[:, :, 0] + pan_2ch[:, :, 1]
            pan = OFFSET * pan_2ch[:, :, 0] + pan_2ch[:, :, 2]
            pan_format = np.zeros((pan_2ch.shape[0], pan_2ch.shape[1], 3), dtype=np.uint8)
            l = np.unique(pan)
            segm_info = []
            for el in l:
                sem = el // OFFSET
                if sem == VOID:
                    continue
                mask = pan == el
                #### viper class mapping
                # sem = np.int64(viper_new2old[sem])
                #### handling used color for inst id
                if el % OFFSET > 0:
                # if el > OFFSET:
                    # things class
                    if el in inst2color:
                        color = inst2color[el]
                    else:
                        color = color_gererator.get_color(sem)
                        inst2color[el] = color
                else:
                    # stuff class
                    color = color_gererator.get_color(sem)
                pan_format[mask] = color
                index = np.where(mask)
                x = index[1].min()
                y = index[0].min()
                width = index[1].max() - x
                height = index[0].max() - y
                segm_info.append({"category_id": sem.item(), "iscrowd": 0, "id": int(rgb2id(color)), "bbox": [x.item(), y.item(), width.item(), height.item()], "area": mask.sum().item()})
            
            annotations.append({"segments_info": segm_info})
            pan_all.append(pan_format)

        return annotations, pan_all


    def get_unified_pan_result(self, segs, pans, cls_inds, obj_ids=None, stuff_area_limit=4 * 64 * 64, names=None):
        if obj_ids is None:
            obj_ids = [None for _ in range(len(cls_inds))]
        pred_pans_2ch = {}
        figs = []
        max_oid = 100
        for (seg, pan, cls_ind, obj_id, name) in zip(segs, pans, cls_inds, obj_ids, names):
            # handle redundant obj_ids
            if obj_id is not None:
                oid_unique, oid_cnt = np.unique(obj_id, return_counts=True)
                obj_id_ = obj_id[::-1].copy()
                if np.any(oid_cnt > 1): 

                    redundants = oid_unique[oid_cnt>1]
                    for red in redundants:
                        part = obj_id[obj_id==red]
                        for i in range(1,len(part)):
                            part[i]=max_oid
                            max_oid+=1
                        obj_id_[obj_id_==red] = part
                    obj_id = obj_id_[::-1]

            pan_seg = pan.copy()
            pan_ins = pan.copy()
            pan_obj = pan.copy()
            id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes
            ids = np.unique(pan)
            ids_ins = ids[ids > id_last_stuff]
            pan_ins[pan_ins <= id_last_stuff] = 0
            for idx, id in enumerate(ids_ins):
                region = (pan_ins == id)
                if id == 255:
                    pan_seg[region] = 255
                    pan_ins[region] = 0
                    continue
                cls, cnt = np.unique(seg[region], return_counts=True)
                if cls[np.argmax(cnt)] == cls_ind[id - id_last_stuff - 1] + id_last_stuff:
                    pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                    pan_ins[region] = idx + 1
                    if obj_id is not None:
                        pan_obj[region] = obj_id[idx] + 1
                else:
                    if np.max(cnt) / np.sum(cnt) >= 0.5 and cls[np.argmax(cnt)] <= id_last_stuff:
                        pan_seg[region] = cls[np.argmax(cnt)]
                        pan_ins[region] = 0 
                        pan_obj[region] = 0
                    else:
                        pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
                        pan_ins[region] = idx + 1
                        if obj_id is not None:
                            pan_obj[region] = obj_id[idx] + 1

            idx_sem = np.unique(pan_seg)
            for i in range(idx_sem.shape[0]):
                if idx_sem[i] <= id_last_stuff:
                    area = pan_seg == idx_sem[i]
                    if (area).sum() < stuff_area_limit:
                        pan_seg[area] = 255

            pan_2ch = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.uint8)
            pan_2ch[:, :, 0] = pan_seg
            pan_2ch[:, :, 1] = pan_ins
            pan_2ch[:, :, 2] = pan_obj
            #### DEBUG
            # import matplotlib
            # matplotlib.use("TkAgg")
            # import matplotlib.pyplot as plt
            # if name in ['011_00035.jpg', '011_00036.jpg']:
            #     tmp = pan_seg.copy()
            #     tmp[tmp==255]=0
            #     figs.append(tmp)
            #     figs.append(pan_ins.copy())
            #     figs.append(pan_obj.copy())
            # if name == '011_00036.jpg':
            #     plt.subplot(231),plt.imshow(figs[0])
            #     plt.subplot(232),plt.imshow(figs[1])
            #     plt.subplot(233),plt.imshow(figs[2])
            #     plt.subplot(234),plt.imshow(figs[3])
            #     plt.subplot(235),plt.imshow(figs[4])
            #     plt.subplot(236),plt.imshow(figs[5])
            #     plt.show()
            #     pdb.set_trace()

            # pred_pans_2ch.append(pan_2ch)
            pred_pans_2ch[name]=pan_2ch
        return pred_pans_2ch


    # def get_tracked_pan_result(self, segs, pans, cls_inds, obj_ids, stuff_area_limit=4 * 64 * 64, names=None):
    #     pred_pans_2ch = {}
    #     for (seg, pan, cls_ind, obj_id, name) in zip(segs, pans, cls_inds, obj_ids, names):
    #         assert len(cls_inds) == len(obj_ids)
    #         pan_seg = pan.copy()
    #         pan_ins = pan.copy()
    #         id_last_stuff = config.dataset.num_seg_classes - config.dataset.num_classes
    #         ids = np.unique(pan)
    #         ids_ins = ids[ids > id_last_stuff]
    #         pan_ins[pan_ins <= id_last_stuff] = 0
    #         # if name in ['011_00035.jpg','011_00036.jpg']:
    #         #     pdb.set_trace()
    #         for idx, id in enumerate(ids_ins):
    #             region = (pan_ins == id)
    #             if id == 255:
    #                 pan_seg[region] = 255
    #                 pan_ins[region] = 0
    #                 continue
    #             cls, cnt = np.unique(seg[region], return_counts=True)
    #             if cls[np.argmax(cnt)] == cls_ind[id - id_last_stuff - 1] + id_last_stuff:
    #                 pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
    #                 # pan_ins[region] = idx + 1
    #                 pan_ins[region] = obj_id[id - id_last_stuff - 1] + 1
    #             else:
    #                 if np.max(cnt) / np.sum(cnt) >= 0.5 and cls[np.argmax(cnt)] <= id_last_stuff:
    #                     pan_seg[region] = cls[np.argmax(cnt)]
    #                     pan_ins[region] = 0 
    #                 else:
    #                     pan_seg[region] = cls_ind[id - id_last_stuff - 1] + id_last_stuff
    #                     # pan_ins[region] = idx + 1
    #                     pan_ins[region] = obj_id[id - id_last_stuff - 1] + 1

    #         idx_sem = np.unique(pan_seg)
    #         for i in range(idx_sem.shape[0]):
    #             if idx_sem[i] <= id_last_stuff:
    #                 area = pan_seg == idx_sem[i]
    #                 if (area).sum() < stuff_area_limit:
    #                     pan_seg[area] = 255

    #         pan_2ch = np.zeros((pan.shape[0], pan.shape[1], 3), dtype=np.uint8)
    #         pan_2ch[:, :, 0] = pan_seg
    #         pan_2ch[:, :, 1] = pan_ins
    #         # pred_pans_2ch.append(pan_2ch)
    #         pred_pans_2ch[name]=pan_2ch
    #     return pred_pans_2ch