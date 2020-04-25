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
from PIL import Image, ImageDraw
from collections import defaultdict, Sequence

from tools.config.config import config
# from tools.dataset.json_dataset import JsonDataset, extend_with_flipped_entries, filter_for_training, add_bbox_regression_targets
from tools.dataset.json_dataset import JsonDataset
from tools.dataset.base_dataset import BaseDataset
# from upsnet.rpn.assign_anchor import add_rpn_blobs
# from upsnet.bbox.sample_rois import sample_rois
from lib.utils.logging import logger
import pycocotools.mask as mask_util
import pdb

class Cityscapes(BaseDataset):

    def __init__(self, image_sets, flip=False, phase='train', result_path=''):

        super(Cityscapes, self).__init__()
        # self.image_dirs = {
        #     'train': os.path.join(config.dataset.dataset_path, 'images'),
        #     'val': os.path.join(config.dataset.dataset_path, 'images'),
        #     'test': os.path.join(config.dataset.dataset_path, 'images'),
        #     'train_extra': os.path.join(config.dataset.dataset_path, 'images'),
        #     'debug': os.path.join(config.dataset.dataset_path, 'images'),
        # }

        # self.anno_files = {
        #     'train': 'instancesonly_gtFine_train.json',
        #     'val': 'instancesonly_gtFine_val.json',
        #     'test': 'image_info_test.json',
        #     'train_extra': 'instancesonly_gtCoarse_train_extra.json',
        #     'debug': 'instancesonly_gtFine_debug.json',
        # }
        self.image_dirs = {
            'train': os.path.join(config.dataset.dataset_path, 'images'),
            'val': os.path.join(config.dataset.dataset_path, 'images'),
            'test': os.path.join(config.dataset.dataset_path, 'images'),
            # 'train_extra': os.path.join(config.dataset.dataset_path, 'images'),
            # 'debug': os.path.join(config.dataset.dataset_path, 'images'),
        }

        self.anno_files = {
            'train': 'instancesonly_gtFine_train.json',
            'val': 'instancesonly_gtFine_val.json',
            'test': 'image_info_test.json',
            # 'train_extra': 'instancesonly_gtCoarse_train_extra.json',
            # 'debug': 'instancesonly_gtFine_debug.json',
        }

        self.panoptic_json_file = os.path.join(
                config.dataset.dataset_path, 
                'annotations', 'cityscapes_fine_val.json')
        self.panoptic_gt_folder = 'data/cityscapes/panoptic'

        self.flip = flip
        self.result_path = result_path
        self.phase = phase
        self.image_sets = image_sets

        # if proposal_files is None:
        #     proposal_files = [None] * len(image_sets)

        # test dataset
        assert len(image_sets) == 1
        self.dataset = JsonDataset('cityscapes_' + image_sets[0],
                            image_dir=self.image_dirs[image_sets[0]],
                            anno_file=os.path.join(config.dataset.dataset_path, 'annotations',
                            self.anno_files[image_sets[0]]))

        roidb = self.dataset.get_roidb()
        self.roidb = roidb

    
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
        self.write_segmentation_result(pred_segmentations, 
                                       res_file_folder,
                                       pred_segm_names)

        confusion_matrix = np.zeros((config.dataset.num_seg_classes, 
                config.dataset.num_seg_classes))

        for i, roidb in enumerate(self.roidb):

            seg_gt = np.array(Image.open(roidb['image'].replace('images', 'labels').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png'))).astype('float32')
            seg_pathes = os.path.split(roidb['image'].replace('images', 'labels').replace('leftImg8bit.png', 'gtFine_labelTrainIds.png'))
            res_image_name = seg_pathes[-1][:-len('_gtFine_labelTrainIds.png')]
            res_save_path = os.path.join(res_file_folder, res_image_name + '.png')
            seg_pred = Image.open(res_save_path)

            seg_pred = np.array(seg_pred.resize((seg_gt.shape[1], seg_gt.shape[0]), Image.NEAREST))
            ignore_index = seg_gt != 255
            seg_gt = seg_gt[ignore_index]
            seg_pred = seg_pred[ignore_index]

            confusion_matrix += self.get_confusion_matrix(
                    seg_gt, seg_pred, 
                    config.dataset.num_seg_classes)

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)

        IU_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_IU = IU_array.mean()

        evaluation_results = {'meanIU':mean_IU, 
                              'IU_array':IU_array, 
                              'confusion_matrix':confusion_matrix}

        def convert_confusion_matrix(confusion_matrix):
            cls_sum = confusion_matrix.sum(axis=1)
            confusion_matrix = confusion_matrix / cls_sum.reshape((-1, 1))
            return confusion_matrix

        # logger.info('evaluate segmentation:')
        print('evaluate segmentation:')
        meanIU = evaluation_results['meanIU']
        IU_array = evaluation_results['IU_array']
        confusion_matrix = convert_confusion_matrix(
                evaluation_results['confusion_matrix'])
        # logger.info('IU_array:')
        print('IU_array:')
        for i in range(len(IU_array)):
            # logger.info('%.5f' % IU_array[i])
            print('%.5f' % IU_array[i])
        # logger.info('meanIU:%.5f' % meanIU)
        print('meanIU:%.5f' % meanIU)
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        import re
        confusion_matrix = re.sub('[\[\]]', '', np.array2string(confusion_matrix, separator='\t'))
        # logger.info('confusion_matrix:')
        # logger.info(confusion_matrix)
        print('confusion_matrix:')
        print(confusion_matrix)

    def write_segmentation_result(self, segmentation_results,
                                  res_file_folder, pred_segm_names):
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

            res_save_path = os.path.join(res_file_folder, pred_segm_name.replace('_leftImg8bit.png','.png')).replace('_newImg8bit.png', '.png')
            segmentation_result = np.uint8(np.squeeze(np.copy(segmentation_results[i])))

            segmentation_result = Image.fromarray(segmentation_result)
            segmentation_result.putpalette(pallete)
            segmentation_result.save(res_save_path)

