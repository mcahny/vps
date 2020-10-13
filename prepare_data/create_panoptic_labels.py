import datetime
import json
import os, sys
import os.path as osp
from PIL import Image
import numpy as np
sys.path.append('../')
from city_default import CATEGORIES, INFO, LICENSES
import argparse
import shutil
import time
import multiprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='val', help='train/val/test')
parser.add_argument('--root_dir', type=str, default='data/city_ext/', help='root directory')
args = parser.parse_args()

MODE = args.mode
ROOT_DIR = args.root_dir
SEMANTIC_DIR = os.path.join(ROOT_DIR, MODE, 'cls')
INSTANCE_DIR = os.path.join(ROOT_DIR, MODE, 'inst')
LABELMAP_DIR = os.path.join(ROOT_DIR, MODE, 'labelmap')
os.makedirs(LABELMAP_DIR, exist_ok=True)
PANOPTIC_DIR = os.path.join(ROOT_DIR, MODE, 'panoptic_inst')
os.makedirs(PANOPTIC_DIR, exist_ok=True)

####
VOID = 255
ID2CATINFO = {x['id']:x for x in CATEGORIES}

def panoptic_single_core(proc_id, sem_files, inst_files, id_converter, ori2fcn):
    for sem_file, inst_file in zip(sem_files, inst_files):
        pan_map, label_map = sem_inst2pan(sem_file, inst_file, id_converter, ori2fcn)
        Image.fromarray(pan_map).save(
            os.path.join(PANOPTIC_DIR, sem_file.split('/')[-1]))
        Image.fromarray(label_map).save(
            os.path.join(LABELMAP_DIR, sem_file.split('/')[-1]))

# convert into Cityscapes-style instanceID maps
def sem_inst2pan(sem_file, inst_file, id_converter, ori2fcn):
    color_map = np.array(Image.open(sem_file), dtype=np.uint32)[:,:,:3] # 3 channel
    sem_map = color_map[:,:,0] + \
                color_map[:,:,1]*256 + \
                color_map[:,:,2]*256*256 # 1 channel
    
    inst_map = np.array(Image.open(inst_file))
    pan_map = np.ones((sem_map.shape[0], sem_map.shape[1]), dtype=np.uint32)*VOID
    label_map = np.ones((sem_map.shape[0], sem_map.shape[1]), dtype=np.uint8)*VOID

    sem_ids = np.unique(sem_map)
    inst_ids = np.unique(inst_map)

    # Stuff Classes  
    for sem_id in sem_ids:
        if sem_id not in id_converter:
            continue
        fcn_id = id_converter[sem_id]
        mask = sem_map == sem_id
        label_map[mask] = fcn_id
        # Dont include Things classes in pan_map.
        if ID2CATINFO[fcn_id]['isthing'] == 1:
            continue
        pan_map[mask] = fcn_id

    # Things Classes
    for inst_id in inst_ids:
        #### if stuff classes < 1000, skip
        if inst_id < 1000:
            continue
        obj_mask = inst_map == inst_id
        sem_id, cnt = np.unique(sem_map[obj_mask], return_counts=True)
        sem_id = sem_id[np.argmax(cnt)]
        if sem_id not in id_converter:
            continue
        # Skip the stuff classes
        fcn_id = id_converter[sem_id]
        obj_id = inst_id % 1000 # NOTE: obj_id can be ZERO !
        # Filter Stuff classes
        if ID2CATINFO[fcn_id]['isthing'] == 0:
            continue
        pan_map[obj_mask] = fcn_id*1000 + obj_id

    return pan_map.astype(np.uint32), label_map.astype(np.uint8)

def panoptic_multi_core(sem_files, inst_files, id_converter, ori2fcn):
    cpu_num = multiprocessing.cpu_count()//2
    sem_split = np.array_split(list(sem_files), cpu_num)
    inst_split = np.array_split(list(inst_files), cpu_num)
    assert(len(sem_split) == len(inst_split))
    print("Number of cores: %d, images per core: %d"%
        (cpu_num, len(sem_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []

    for proc_id, (sem_part, inst_part) in enumerate(zip(sem_split, inst_split)):
        workers.apply_async(panoptic_single_core,
                (proc_id, sem_part, inst_part, id_converter, ori2fcn))
    workers.close()
    workers.join()


def main():
    
    id_converter = {x['color'][0]+x['color'][1]*256+x['color'][2]*256*256:x['id'] for x in CATEGORIES}
    ori2fcn = {x['ori_id'] : x['id'] for x in CATEGORIES}
    # panoptic video annotations
    start_pano = time.time()
    print('==> %s/labelmap/, %s/panoptic_inst/ ...'%(MODE, MODE))  
    # merge semantic segmentation and instance id map into panoptic format
    sem_files = [osp.join(SEMANTIC_DIR,x) for x in os.listdir(SEMANTIC_DIR) if '.png' in x]
    sem_files.sort()
    inst_files = [osp.join(INSTANCE_DIR,x) for x in os.listdir(INSTANCE_DIR) if '.png' in x]
    inst_files.sort()
    if not (len(sem_files) == len(inst_files)):
        raise ValueError('len semfiles != len inst_files')
    panoptic_multi_core(sem_files, inst_files, id_converter, ori2fcn)          

if __name__=='__main__':
    main()