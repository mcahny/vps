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
# -----------------------------------------------------------------------
from __future__ import print_function

import os
import os.path as osp
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
from pycocotools.cocoeval import COCOeval

from tools.config.config import config
# from tools.dataset.json_dataset import JsonDataset
from tools.dataset.base_dataset import BaseDataset, PQStat, PQStatCat
# from tools.dataset.base_dataset import PQStat, PQStatCat

from lib.utils.logging import logger
import pycocotools.mask as mask_util
import pdb


class CityscapesExt(BaseDataset):
# class CityscapesExt(torch.utils.data.Dataset):

    def __init__(self):

        super(CityscapesExt, self).__init__()

        self.nframes_per_video=6
        self.lambda_=5
        self.labeled_fid=20



        # **** TRY TO REMOVE THIS PART BELOW ****
        # config.dataset.dataset_path = 'data/cityscapes_ext/'
        # config.dataset.name = 'val'

        # self.image_dirs = {
        #     'train': os.path.join(config.dataset.dataset_path, 'train/img'),
        #     'val': os.path.join(config.dataset.dataset_path, 'val/img'),
        #     'test': os.path.join(config.dataset.dataset_path, 'test/img'),
        # }
        # self.anno_files = {
        #     'train': 'instances_train_01_city_coco_rle.json',
        #     'val': 'instances_val_01_city_coco_rle.json',
        #     'test': 'instances_val_01_city_coco_rle.json',
        # }
        # self.panoptic_json_file = '/data2/video_panoptic/data/cityscapes_ext/cityscapes_ext_panoptic_val_video.json'
        # self.panoptic_gt_folder = '/data2/video_panoptic/data/cityscapes_ext/val/panoptic_video_vivid/'

        # self.flip = flip
        # self.result_path = result_path
        # self.num_classes = 9
        # self.phase = phase
        # self.image_sets = image_sets

        # assert len(image_sets) == 1
        # self.dataset = JsonDataset('cityscapes_ext' + image_sets[0],
        #                     image_dir=self.image_dirs[image_sets[0]],
        #                     anno_file=osp.join(
        #                             config.dataset.dataset_path,
        #                             self.anno_files[image_sets[0]]))

        # roidb = self.dataset.get_roidb()
        # self.roidb = roidb

    def inference_panoptic_video(self, pred_pans_2ch, output_dir, 
                                 # pan_im_json_file,
                                 categories,
                                 names,
                                 n_video=0):
        from panopticapi.utils import IdGenerator
        # pred_pans_2ch = pred_pans_2ch[(self.labeled_fid//self.lambda_)::self.lambda_]
        # with open(pan_im_json_file,'r') as f:
        #     im_jsons = json.load(f)

        # categories = im_jsons['categories']
        categories = {el['id']: el for el in categories}
        color_generator = IdGenerator(categories)

        # def get_pred_large(pan_2ch_all, vid_num, nframes_per_video=6):
            
        #     cpu_num = multiprocessing.cpu_count()
        #     pan_2ch_split = np.array_split(pan_2ch_all, cpu_num)
        #     workers = multiprocessing.Pool(processes=cpu_num)
        #     processes = []
        #     for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
        #         p = workers.apply_async(self.converter_2ch_track_core, (proc_id, pan_2ch_set, color_generator))
        #         processes.append(p)
        #     workers.close()
        #     workers.join()
        #     annotations, pan_all = [], []
        #     for p in processes:
        #         p = p.get()
        #         annotations.extend(p[0])
        #         pan_all.extend(p[1])
        #     pan_json = {'annotations': annotations}
        #     return pan_all, pan_json


        def get_pred_large(pan_2ch_all, vid_num, nframes_per_video=6):
            vid_num = len(pan_2ch_all)//nframes_per_video # 10
            cpu_num = multiprocessing.cpu_count()//2 # 32 --> 16
            nprocs = min(vid_num, cpu_num) # 10
            max_nframes= cpu_num*nframes_per_video
            nsplits = (len(pan_2ch_all)-1)//max_nframes + 1
            annotations, pan_all = [], []
            for i in range(0,len(pan_2ch_all), max_nframes):
                print('==> Read and convert VPS output - split %d/%d'%((i//max_nframes)+1, nsplits))
                pan_2ch_part = pan_2ch_all[i:min(
                        i+max_nframes, len(pan_2ch_all))]
                pan_2ch_split = np.array_split(pan_2ch_part, nprocs)
                workers = multiprocessing.Pool(processes=nprocs)
                processes = []
                for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
                    p = workers.apply_async(
                        self.converter_2ch_track_core, 
                        (proc_id, pan_2ch_set, color_generator))
                    processes.append(p)
                workers.close()
                workers.join()

                for p in processes:
                    p = p.get()
                    annotations.extend(p[0])
                    pan_all.extend(p[1])

            pan_json = {'annotations': annotations}
            return pan_all, pan_json

        def save_image(images, save_folder, names, colors=None):
            os.makedirs(save_folder, exist_ok=True)
            # names = [osp.join(save_folder, item['file_name'].replace('_leftImg8bit', '').replace('_newImg8bit','').replace('jpg', 'png').replace('jpeg', 'png')) for item in im_jsons['images']]
            names = [osp.join(save_folder, name.replace('_leftImg8bit', '').replace('_newImg8bit','').replace('jpg', 'png').replace('jpeg', 'png')) for name in names]
            cpu_num = multiprocessing.cpu_count()//2
            images_split = np.array_split(images, cpu_num)
            names_split = np.array_split(names, cpu_num)
            workers = multiprocessing.Pool(processes=cpu_num)
            for proc_id, (images_set, names_set) in enumerate(zip(images_split, names_split)):
                workers.apply_async(BaseDataset._save_image_single_core, (proc_id, images_set, names_set, colors))
            workers.close()
            workers.join()

        # inference_panoptic_video
        pred_pans, pred_json = get_pred_large(pred_pans_2ch, 
                vid_num=n_video)
        print('--------------------------------------')    
        print('==> Saving VPS output png files')
        os.makedirs(output_dir, exist_ok=True)
        save_image(pred_pans_2ch, osp.join(output_dir, 'pan_2ch'), names)
        save_image(pred_pans, osp.join(output_dir, 'pan_pred'), names)
        print('==> Saving pred.jsons file')
        json.dump(pred_json, open(osp.join(output_dir, 'pred.json'), 'w'))
        print('--------------------------------------') 

        return pred_pans, pred_json


    
    # def evaluate_panoptic_video(self, pred_pans_2ch, output_dir, pan_gt_json_file, pan_gt_folder, 
    #     n_video=0, save_name=None):
    def evaluate_panoptic_video(self, pred_pans, pred_json, 
        output_dir, pan_gt_json_file, pan_gt_folder, 
        n_video=0):
        # sys.path.insert(0, osp.join(osp.abspath(
        #     osp.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))

        from panopticapi.utils import IdGenerator
        # # cityscapes_vps - sample only annotated frames
        # pred_pans_2ch = pred_pans_2ch[(self.labeled_fid//self.lambda_)::self.lambda_]

        def get_gt(pan_gt_json_file, pan_gt_folder):
            with open(pan_gt_json_file, 'r') as f:
                pan_gt_json = json.load(f)
            files = [item['file_name'].replace('leftImg8bit','gtFine_color').replace('newImg8bit','final_mask') for item in pan_gt_json['images']]
            cpu_num = multiprocessing.cpu_count()//2
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
            color_generator = IdGenerator(categories)
            return pan_gt_all, pan_gt_json, categories, color_generator


        # def get_pred_large(pan_2ch_all, color_gerenator, vid_num=100, nframes_per_video=6):

        #     vid_num = len(pan_2ch_all)//nframes_per_video # 10
        #     cpu_num = multiprocessing.cpu_count()//2 # 32 --> 16
        #     nprocs = min(vid_num, cpu_num) # 10
        #     max_nframes= cpu_num*nframes_per_video
        #     nsplits = (len(pan_2ch_all)-1)//max_nframes + 1

        #     annotations, pan_all = [], []
        #     for i in range(0,len(pan_2ch_all), max_nframes):
        #         print('==> Read and convert VPS output - split %d/%d'%((i//max_nframes)+1, nsplits))
        #         pan_2ch_part = pan_2ch_all[i:min(
        #                 i+max_nframes, len(pan_2ch_all))]
        #         pan_2ch_split = np.array_split(pan_2ch_part, nprocs)
        #         workers = multiprocessing.Pool(processes=nprocs)
        #         processes = []
        #         for proc_id, pan_2ch_set in enumerate(pan_2ch_split):
        #             p = workers.apply_async(
        #                 self.converter_2ch_track_core, 
        #                 (proc_id, pan_2ch_set, color_generator))
        #             processes.append(p)
        #         workers.close()
        #         workers.join()

        #         for p in processes:
        #             p = p.get()
        #             annotations.extend(p[0])
        #             pan_all.extend(p[1])

        #     pan_json = {'annotations': annotations}
        #     return pan_all, pan_json


        # def save_image(images, save_folder, gt_json, colors=None):
        #     os.makedirs(save_folder, exist_ok=True)
        #     names = [osp.join(save_folder, item['file_name'].replace('_leftImg8bit', '').replace('_newImg8bit','').replace('jpg', 'png').replace('jpeg', 'png')) for item in gt_json['images']]
        #     cpu_num = multiprocessing.cpu_count()//2
        #     images_split = np.array_split(images, cpu_num)
        #     names_split = np.array_split(names, cpu_num)
        #     workers = multiprocessing.Pool(processes=cpu_num)
        #     for proc_id, (images_set, names_set) in enumerate(zip(images_split, names_split)):
        #         workers.apply_async(BaseDataset._save_image_single_core, (proc_id, images_set, names_set, colors))
        #     workers.close()
        #     workers.join()

            
        def vpq_compute(gt_pred_split, categories, nframes, output_dir):
            start_time = time.time()
            vpq_stat = PQStat()
            for idx, gt_pred_set in enumerate(gt_pred_split):
                tmp = vpq_compute_single_core(gt_pred_set, categories, nframes=nframes)
                vpq_stat += tmp
            print('==> %d-frame vpq_stat:'%(nframes), time.time()-start_time)

            metrics = [("All", None), ("Things", True), ("Stuff", False)]
            results = {}
            for name, isthing in metrics:
                results[name], per_class_results = vpq_stat.pq_average(categories, isthing=isthing)
                if name == 'All':
                    results['per_class'] = per_class_results

            vpq_all = 100 * results['All']['pq']
            vpq_thing = 100 * results['Things']['pq']
            vpq_stuff = 100 * results['Stuff']['pq']

            save_name = os.path.join(output_dir, 'vpq-%d.txt'%(nframes))
            f = open(save_name, 'w') if save_name else None
            f.write("================================================\n")
            f.write("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N\n"))
            f.write("-" * (10 + 7 * 4)+'\n')
            for name, _isthing in metrics:
                f.write("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}\n".format(name, 100 * results[name]['pq'], 100 * results[name]['sq'], 100 * results[name]['rq'], results[name]['n']))
            f.write("{:4s}| {:>5s} {:>5s} {:>5s} {:>6s} {:>7s} {:>7s} {:>7s}\n".format("IDX", "PQ", "SQ", "RQ", "IoU", "TP", "FP", "FN"))
            for idx, result in results['per_class'].items():
                f.write("{:4d} | {:5.1f} {:5.1f} {:5.1f} {:6.1f} {:7d} {:7d} {:7d}\n".format(idx, 100 * result['pq'], 100 * result['sq'], 100 * result['rq'], result['iou'], result['tp'], result['fp'], result['fn']))
            if save_name:
                f.close()

            return vpq_all, vpq_thing, vpq_stuff
            
        #### other wise:
        gt_pans, gt_jsons, categories, color_generator = get_gt(pan_gt_json_file, pan_gt_folder)
        # pred_pans, pred_json = get_pred_large(pred_pans_2ch,
        #         color_generator, vid_num=n_video)

        # print('--------------------------------------')    
        # print('==> Saving VPS output png files')
        # print('--------------------------------------') 
        # os.makedirs(output_dir, exist_ok=True)
        # save_image(pred_pans_2ch, osp.join(output_dir, 'pan_2ch'), gt_json)
        # save_image(pred_pans, osp.join(output_dir, 'pan'), gt_json)
        # # # pdb.set_trace()
        # json.dump(gt_json, open(osp.join(output_dir, 'gt.json'), 'w'))
        # json.dump(pred_json, open(osp.join(output_dir, 'pred.json'), 'w'))

        # from json and from numpy
        gt_image_jsons = gt_jsons['images']
        gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
        gt_pred_all = list(zip(gt_jsons, pred_jsons, gt_pans, pred_pans,
                               gt_image_jsons))
        gt_pred_split = np.array_split(gt_pred_all, n_video)
        vpq_all, vpq_thing, vpq_stuff = [], [], []

        # k = [0, 5, 10, 15]
        for nframes in [1,2,3,4]:
            gt_pred_split_ = copy.deepcopy(gt_pred_split)
            vpq_all_, vpq_thing_, vpq_stuff_ = vpq_compute(
                    gt_pred_split_, categories, nframes, output_dir)
            del gt_pred_split_
            print(vpq_all_, vpq_thing_, vpq_stuff_)
            vpq_all.append(vpq_all_)
            vpq_thing.append(vpq_thing_)
            vpq_stuff.append(vpq_stuff_)

        return vpq_all, vpq_thing, vpq_stuff

    def vpq_compute_single_core(gt_pred_set, categories, nframes=2):
        OFFSET = 256 * 256 * 256
        VOID = 0
        vpq_stat = PQStat()
        # nframes=2
        # gt_pred_set: len=6
        # idx: 0,1,2,3,4
        # start_time = time.time()
        # for idx in range(0, len(gt_pans_set)-nframes+1): 
        for idx in range(0, len(gt_pred_set)-nframes+1): 

            vid_pan_gt, vid_pan_pred = [], []
            gt_segms_list, pred_segms_list = [], []

            #### Output VPQ value for "nframe" volume.
            # Step1. to merge jsons and pan maps in the volume.
            # for gt_json, pred_json, gt_pan, pred_pan, gt_image_json in vid_pan_set:
            for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(gt_pred_set[idx:idx+nframes]):
                
                gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
                pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
                pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256
                # gt_segms = {el['id']: el for el in gt_json['segments_info']}
                gt_segms = {}
                for el in gt_json['segments_info']:
                    if el['id'] in gt_segms:
                        gt_segms[el['id']]['area'] += el['area']
                    else:
                        gt_segms[el['id']] = el

                # pred_segms = {el['id']: el for el in pred_json['segments_info']}
                pred_segms = {}
                for el in pred_json['segments_info']:
                    if el['id'] in pred_segms:
                        pred_segms[el['id']]['area'] += el['area']
                    else:
                        pred_segms[el['id']] = el

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

            # Step 2. stack the collected elements ==> tube-level 
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
            # gt_small = set()
            gt_matched = set()
            pred_matched = set()
            tp = 0
            fp = 0
            fn = 0

            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                # pred_area = (vid_pan_pred == pred_label).sum()
                # gt_area = (vid_pan_gt == gt_label).sum()

                if gt_label not in vid_gt_segms:
                    continue
                if pred_label not in vid_pred_segms:
                    continue
                if vid_gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if vid_gt_segms[gt_label]['category_id'] != \
                        vid_pred_segms[pred_label]['category_id']:
                    continue

                union = vid_pred_segms[pred_label]['area'] + vid_gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                    (VOID, pred_label), 0)
                iou = intersection / union

                # ignore invalid iou value
                assert iou <= 1.0, 'INVALID IOU VALUE : %d'%(gt_label)

                if iou > 0.5:
                    vpq_stat[vid_gt_segms[gt_label]['category_id']].tp += 1
                    vpq_stat[vid_gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)
                    tp += 1

            # count false negatives
            crowd_labels_dict = {}
            for gt_label, gt_info in vid_gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                # if gt_label in gt_small:
                #     continue
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

        return vpq_stat


    def converter_2ch_track_core(self, proc_id, pan_2ch_set, color_generator):
        # sys.path.insert(0, osp.join(osp.abspath(osp.dirname(__file__)), '..', '..', 'lib', 'dataset_devkit'))
        from panopticapi.utils import rgb2id

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
            # segm_info = []
            segm_info = {}
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
                        color = color_generator.get_color(sem)
                        inst2color[el] = color
                else:
                    # stuff class
                    color = color_generator.get_color(sem)

                pan_format[mask] = color
                index = np.where(mask)
                x = index[1].min()
                y = index[0].min()
                width = index[1].max() - x
                height = index[0].max() - y
                # segm_info.append({"category_id": sem.item(), "iscrowd": 0, "id": int(rgb2id(color)), "bbox": [x.item(), y.item(), width.item(), height.item()], "area": mask.sum().item()})
                dt = {"category_id": sem.item(), "iscrowd": 0, "id": int(rgb2id(color)), "bbox": [x.item(), y.item(), width.item(), height.item()], "area": mask.sum().item()}
                segment_id = int(rgb2id(color))
                segm_info[segment_id] = dt
            
            # annotations.append({"segments_info": segm_info})
            pan_all.append(pan_format)
            
            gt_pan = np.uint32(pan_format)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256      
            labels, labels_cnt = np.unique(pan_gt, return_counts=True)
            for label, area in zip(labels, labels_cnt):
                if label == 0:
                    continue
                if label not in segm_info.keys():
                    print('label:', label)
                    raise KeyError('label not in segm_info keys.')

                segm_info[label]["area"] = int(area)
            segm_info = [v for k,v in segm_info.items()]

            annotations.append({"segments_info": segm_info})

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

            pred_pans_2ch[name]=pan_2ch
        return pred_pans_2ch

