# -------------------------------------------------------------------
# Video Panoptic Segmentation
#
# VPQ evaluation code by tube (video segment) matching
# Inference on every frames and evaluation on every 5 frames.
# ------------------------------------------------------------------

import argparse
import sys
import os
import os.path
import numpy as np
from PIL import Image
import multiprocessing
import time
import json
from collections import defaultdict
import copy
import pdb

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


def vpq_compute_single_core(gt_pred_set, categories, nframes=2):
    OFFSET = 256 * 256 * 256
    VOID = 0
    vpq_stat = PQStat()

    # Iterate over the video frames 0::T-λ
    for idx in range(0, len(gt_pred_set)-nframes+1): 
        vid_pan_gt, vid_pan_pred = [], []
        gt_segms_list, pred_segms_list = [], []

        # Matching nframes-long tubes.
        # Collect tube IoU, TP, FP, FN
        for i, (gt_json, pred_json, gt_pan, pred_pan, gt_image_json) in enumerate(gt_pred_set[idx:idx+nframes]):
            #### Step1. Collect frame-level pan_gt, pan_pred, etc.
            gt_pan, pred_pan = np.uint32(gt_pan), np.uint32(pred_pan)
            pan_gt = gt_pan[:, :, 0] + gt_pan[:, :, 1] * 256 + gt_pan[:, :, 2] * 256 * 256
            pan_pred = pred_pan[:, :, 0] + pred_pan[:, :, 1] * 256 + pred_pan[:, :, 2] * 256 * 256
            gt_segms = {}
            for el in gt_json['segments_info']:
                if el['id'] in gt_segms:
                    gt_segms[el['id']]['area'] += el['area']
                else:
                    gt_segms[el['id']] = copy.deepcopy(el)
            pred_segms = {}
            for el in pred_json['segments_info']:
                if el['id'] in pred_segms:
                    pred_segms[el['id']]['area'] += el['area']
                else:
                    pred_segms[el['id']] = copy.deepcopy(el)
            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in pred_json['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == VOID:
                        continue
                    raise KeyError('Segment with ID {} is presented in PNG and not presented in JSON.'.format(label))
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in categories:
                    raise KeyError('Segment with ID {} has unknown category_id {}.'.format(label, pred_segms[label]['category_id']))
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'The following segment IDs {} are presented in JSON and not presented in PNG.'.format(list(pred_labels_set)))

            vid_pan_gt.append(pan_gt)
            vid_pan_pred.append(pan_pred)
            gt_segms_list.append(gt_segms)
            pred_segms_list.append(pred_segms)

        #### Step 2. Concatenate the collected items -> tube-level. 
        vid_pan_gt = np.stack(vid_pan_gt) # [nf,H,W]
        vid_pan_pred = np.stack(vid_pan_pred) # [nf,H,W]
        vid_gt_segms, vid_pred_segms = {}, {}
        for gt_segms, pred_segms in zip(gt_segms_list, pred_segms_list):
            # aggregate into tube 'area'
            for k in gt_segms.keys():
                if not k in vid_gt_segms:
                    vid_gt_segms[k] = gt_segms[k]
                else:
                    vid_gt_segms[k]['area'] += gt_segms[k]['area']
            for k in pred_segms.keys():
                if not k in vid_pred_segms:
                    vid_pred_segms[k] = pred_segms[k]
                else:
                    vid_pred_segms[k]['area'] += pred_segms[k]['area']

        #### Step3. Confusion matrix calculation
        vid_pan_gt_pred = vid_pan_gt.astype(np.uint64) * OFFSET + vid_pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(vid_pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        tp = 0
        fp = 0
        fn = 0

        #### Step4. Tube matching
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple

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
            assert iou <= 1.0, 'INVALID IOU VALUE : %d'%(gt_label)
            # count true positives
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


def vpq_compute(gt_pred_split, categories, nframes, output_dir):
    start_time = time.time()
    vpq_stat = PQStat()
    for idx, gt_pred_set in enumerate(gt_pred_split):
        tmp = vpq_compute_single_core(gt_pred_set, categories, nframes=nframes)
        vpq_stat += tmp

    # hyperparameter: window size k
    k = (nframes-1)*5
    print('==> %d-frame vpq_stat:'%(k), time.time()-start_time, 'sec')
    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = vpq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results

    vpq_all = 100 * results['All']['pq']
    vpq_thing = 100 * results['Things']['pq']
    vpq_stuff = 100 * results['Stuff']['pq']

    save_name = os.path.join(output_dir, 'vpq-%d.txt'%(k))
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


def parse_args():
    parser = argparse.ArgumentParser(description='VPSNet eval')
    parser.add_argument('--submit_dir', type=str,
        help='test outout directory', default='work_dirs/cityscapes_vps/fusetrack_vpct/val_pans_unified/') 
    parser.add_argument('--truth_dir', type=str, 
        help='ground truth directory', default='data/cityscapes_vps/val/panoptic_video')
    parser.add_argument('--pan_gt_json_file', type=str, 
        help='ground truth directory', default='data/cityscapes_vps/panpotic_gt_val_city_vps.json')    
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    submit_dir = args.submit_dir
    truth_dir = args.truth_dir
    output_dir = submit_dir
    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)
    if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    start_all = time.time()
    pan_pred_json_file = os.path.join(submit_dir, 'pred.json')
    with open(pan_pred_json_file, 'r') as f:
        pred_jsons = json.load(f)
    pan_gt_json_file = args.pan_gt_json_file
    with open(pan_gt_json_file, 'r') as f:
        gt_jsons = json.load(f)

    categories = gt_jsons['categories']
    categories = {el['id']: el for el in categories}
    # ==> pred_json, gt_json, categories

    start_time = time.time()
    gt_pans = []
    files = [item['file_name'].replace('_newImg8bit.png','_final_mask.png').replace('_leftImg8bit.png','_gtFine_color.png') for item in gt_jsons['images']]
    files.sort()
    for idx, file in enumerate(files):
        image = np.array(Image.open(os.path.join(truth_dir, file)))
        gt_pans.append(image)
    print('==> gt_pans:', len(gt_pans), '//', time.time() - start_time,'sec')

    start_time = time.time()
    pred_pans = []
    files = [item['id']+'.png' for item in gt_jsons['images']]
    for idx, file in enumerate(files):
        image = np.array(Image.open(os.path.join(submit_dir, 'pan_pred', file)))
        pred_pans.append(image)
    print('==> pred_pans:', len(pred_pans), '//', time.time() - start_time,'sec')
    assert len(gt_pans) == len(pred_pans), "number of prediction does not match with the groud truth."

    gt_image_jsons = gt_jsons['images']
    gt_jsons, pred_jsons = gt_jsons['annotations'], pred_jsons['annotations']
    nframes_per_video = 6
    vid_num = len(gt_jsons)//nframes_per_video # 600//6 = 100

    gt_pred_all = list(zip(gt_jsons, pred_jsons, gt_pans, pred_pans, gt_image_jsons))
    gt_pred_split = np.array_split(gt_pred_all, vid_num)

    start_time = time.time()
    vpq_all, vpq_thing, vpq_stuff = [], [], []

    # for k in [0,5,10,15] --> num_frames_w_gt [1,2,3,4]
    for nframes in [1,2,3,4]:
        gt_pred_split_ = copy.deepcopy(gt_pred_split)
        vpq_all_, vpq_thing_, vpq_stuff_ = vpq_compute(
                gt_pred_split_, categories, nframes, output_dir)
        del gt_pred_split_
        print(vpq_all_, vpq_thing_, vpq_stuff_)
        vpq_all.append(vpq_all_)
        vpq_thing.append(vpq_thing_)
        vpq_stuff.append(vpq_stuff_)

    output_filename = os.path.join(output_dir, 'vpq-final.txt')
    output_file = open(output_filename, 'w')
    output_file.write("vpq_all:%.4f\n"%(sum(vpq_all)/len(vpq_all)))
    output_file.write("vpq_thing:%.4f\n"%(sum(vpq_thing)/len(vpq_thing)))
    output_file.write("vpq_stuff:%.4f\n"%(sum(vpq_stuff)/len(vpq_stuff)))
    output_file.close()
    print('==> All:', time.time() - start_all, 'sec')


if __name__ == "__main__":
    main()