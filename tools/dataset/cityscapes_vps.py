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
import torch.multiprocessing as multiprocessing
import pickle
import numpy as np
import json
import time
from PIL import Image
from tools.config.config import config
from tools.dataset.base_dataset import BaseDataset


class CityscapesVps(BaseDataset):

    def __init__(self):

        super(CityscapesVps, self).__init__()

        self.nframes_per_video=6
        self.lambda_=5
        self.labeled_fid=20

    def inference_panoptic_video(self, pred_pans_2ch, output_dir, 
                                 # pan_im_json_file,
                                 categories,
                                 names,
                                 n_video=0):
        from panopticapi.utils import IdGenerator
        
        # Sample only frames with GT annotations.
        pred_pans_2ch = pred_pans_2ch[(self.labeled_fid//self.lambda_)::self.lambda_]
        categories = {el['id']: el for el in categories}
        color_generator = IdGenerator(categories)

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


    def converter_2ch_track_core(self, proc_id, pan_2ch_set, color_generator):
        from panopticapi.utils import rgb2id

        OFFSET = 1000
        VOID = 255
        annotations, pan_all = [], []
        # reference dict to used color
        inst2color = {}
        for idx in range(len(pan_2ch_set)):
            pan_2ch = np.uint32(pan_2ch_set[idx])
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

