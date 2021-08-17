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
import os.path as osp
import sys
import torch
import torch.multiprocessing as multiprocessing
import pickle
import numpy as np
import json
import time
from PIL import Image
from tools.config.config import config
from tools.dataset.base_dataset import BaseDataset
import cv2
from PIL import Image, ImageDraw
from collections import defaultdict


# panoptic visualization
vis_panoptic = False

class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self

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
            #### exclude "mobilebarrier"
            if label == 11:
                continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
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
        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results

class Viper(BaseDataset):

    def __init__(self, image_sets=['val'], flip=False, proposal_files=None, phase='test', result_path=''):

        super(Viper, self).__init__()
        #### Re-configuration for Viper dataset
        config.dataset.dataset_path = 'data/viper_vps/'
        config.dataset.num_seg_classes = 23
        config.dataset.num_classes = 11
        config.dataset.name = 'val_day'

        self.image_dirs = {
            'train': os.path.join(config.dataset.dataset_path, config.dataset.name, 'img'),
            'val': os.path.join(config.dataset.dataset_path, config.dataset.name, 'img'),
            'test': os.path.join(config.dataset.dataset_path, config.dataset.name, 'img'),
        }

        self.anno_files = {
            'train': 'instances_train_05_viper_coco.json',
            'val': 'instances_%s_01_viper_coco.json'%(config.dataset.name),
            'test': 'instances_%s_01_viper_coco.json'%(config.dataset.name),
        }

        self.panoptic_json_file = 'data/viper_vps/viper_panoptic_val_day_video.json'
        self.panoptic_gt_folder = 'data/viper_vps/val_day/panoptic_video'

        self.flip = flip
        self.result_path = result_path
        self.num_classes = 9
        self.phase = phase
        self.image_sets = image_sets

        if proposal_files is None:
            proposal_files = [None] * len(image_sets)

    def get_pallete(self):

        pallete_raw = np.zeros((256, 3)).astype('uint8')
        pallete = np.zeros((256, 3)).astype('uint8')
        
        
        pallete_raw[1, :] =  [70,130,180]
        pallete_raw[2, :] =  [128,64,128]
        pallete_raw[3, :] =  [244,35,232]
        pallete_raw[4, :] =  [152,251,152]
        pallete_raw[5, :] =  [87,182,35]
        pallete_raw[6, :] = [35,142,35]
        pallete_raw[7, :] = [ 70,  70,  70]
        pallete_raw[8, :] = [153,153,153]
        pallete_raw[9, :] = [190,153,153]
        pallete_raw[10, :] = [150,20,20]
        pallete_raw[11, :] = [220,220,0]
        pallete_raw[12, :] = [180,180,100]
        pallete_raw[13, :] = [81,0,21]
        pallete_raw[14, :] = [250,170,30]
        pallete_raw[15, :] = [173,153,153]
        pallete_raw[16, :] = [168,153,153]
        pallete_raw[17, :] = [81,0,81]
        pallete_raw[18, :] = [220,20,60]
        pallete_raw[19, :] = [0,0,230]
        pallete_raw[20, :] = [0,0,142]
        pallete_raw[21, :] = [0,80,100]
        pallete_raw[22, :] = [0,60,100]
        pallete_raw[23, :] = [0,0,70]
        # pallete_raw[-1, :] = [0,0,0]
       
        # train2regular = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        train2regular = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
        # train2regular = [0,11,12,13,14,15,16,17,18,19,20,21,22,23,1,2,3,4,5,6,7,8,9,10]

        for i in range(len(train2regular)):
            pallete[i, :] = pallete_raw[train2regular[i], :]

        pallete = pallete.reshape(-1)

        # return pallete_raw
        return pallete

    def evaluate_panoptic(self, pred_pans_2ch, output_dir, n_video=0, save_name=None):
        sys.path.insert(0, os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))

        from panopticapi.utils import IdGenerator

        def get_gt(pan_gt_json_file=None, pan_gt_folder=None):
            if pan_gt_json_file is None:
                pan_gt_json_file = self.panoptic_json_file
            if pan_gt_folder is None:
                pan_gt_folder = self.panoptic_gt_folder
            with open(pan_gt_json_file, 'r') as f:
                pan_gt_json = json.load(f)
            files = [item['file_name'] for item in pan_gt_json['images']]
            if 'viper' in pan_gt_folder:
                files = [_.split('/')[-1].replace('.jpg', '.png') for _ in files]
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

        def save_image(images, save_folder, gt_json, colors=None):
            os.makedirs(save_folder, exist_ok=True)
            names = [os.path.join(save_folder, item['file_name'].replace('_leftImg8bit', '').replace('jpg', 'png').replace('jpeg', 'png')) for item in gt_json['images']]
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
            assert cpu_num >= vid_num, "vid_num must be smaller than cpu_num."
            gt_jsons_split, pred_jsons_split = np.array_split(gt_jsons, vid_num), np.array_split(pred_jsons, vid_num)
            gt_pans_split, pred_pans_split = np.array_split(gt_pans, vid_num), np.array_split(pred_pans, vid_num)
            gt_image_jsons_split = np.array_split(gt_image_jsons, vid_num)

            workers = multiprocessing.Pool(processes=vid_num)
            processes = []
            for proc_id, (gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set) in enumerate(zip(gt_jsons_split, pred_jsons_split, gt_pans_split, pred_pans_split, gt_image_jsons_split)):
                ####****************** DEBUG ******************
                # vpq = Viper._vpq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories, nframes=nframes)
                # pdb.set_trace()
                p = workers.apply_async(Viper._vpq_compute_single_core, (proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories, nframes))
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
                p = workers.apply_async(Viper._pq_compute_single_core, (proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories))
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

        gt_pans, gt_json, categories, color_gererator = get_gt()
        pred_pans, pred_json = get_pred(pred_pans_2ch, color_gererator, 
                                        cpu_num=n_video)
        save_image(pred_pans_2ch, os.path.join(output_dir, 'pan_2ch'), gt_json)
        save_image(pred_pans, os.path.join(output_dir, 'pan'), gt_json)
        
        json.dump(gt_json, open(os.path.join(output_dir, 'gt.json'), 'w'))
        json.dump(pred_json, open(os.path.join(output_dir, 'pred.json'), 'w'))
        results = pq_compute(gt_json, pred_json, gt_pans, pred_pans, categories, save_name=save_name+'_vpq_nf01.txt')
        for nframes in [5,10,15]:
            results = vpq_compute(gt_json, pred_json, gt_pans, pred_pans, categories, vid_num=n_video, nframes=nframes, save_name=save_name+'_vpq_nf%02d.txt'%(nframes))
        return results

    @staticmethod
    def _vpq_compute_single_core(proc_id, gt_jsons_set, pred_jsons_set, gt_pans_set, pred_pans_set, gt_image_jsons_set, categories, nframes=2):
        OFFSET = 256 * 256 * 256
        VOID = 0
        SIZE_THR = 32**2
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
        SIZE_THR = 32**2
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
        viper_new2old = {k:k for k in range(23)}
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
                sem = np.int64(viper_new2old[sem])
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
                # pdb.set_trace()
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
            pred_pans_2ch[name]=pan_2ch
        return pred_pans_2ch