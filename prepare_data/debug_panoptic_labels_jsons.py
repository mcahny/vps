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
parser.add_argument('--root_dir', type=str, default='data/city_dbg_0627/', help='root directory')
args = parser.parse_args()

MODE = args.mode
ROOT_DIR = args.root_dir
assert(os.path.exists(ROOT_DIR))
SEMANTIC_DIR = osp.join(ROOT_DIR, MODE, 'cls')
INSTANCE_DIR = osp.join(ROOT_DIR, MODE, 'inst')
LABELMAP_DIR = osp.join(ROOT_DIR, MODE, 'labelmap')
os.makedirs(LABELMAP_DIR, exist_ok=True)
PANOPTIC_DIR = osp.join(ROOT_DIR, MODE, 'panoptic_inst')
os.makedirs(PANOPTIC_DIR, exist_ok=True)

####
VOID = 255
ID2CATINFO = {x['id']:x for x in CATEGORIES}

class panoptic_creator():
    
    def __init__(self, id_converter, ori2fcn):

        self.id_converter=id_converter
        self.ori2fcn=ori2fcn

    # convert into Cityscapes-style instanceID maps
    def sem_inst2pan(self, sem_file, inst_file):
        print('gothere2')
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
            if sem_id not in self.id_converter:
                continue
            fcn_id = self.id_converter[sem_id]
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
            if sem_id not in self.id_converter:
                continue
            # Skip the stuff classes
            fcn_id = self.id_converter[sem_id]
            obj_id = inst_id % 1000 # NOTE: obj_id can be ZERO !
            # Filter Stuff classes
            if ID2CATINFO[fcn_id]['isthing'] == 0:
                continue
            pan_map[obj_mask] = fcn_id*1000 + obj_id
        # Heuristics for the Ego-Centric Car
        ego_map = label_map[int(label_map.shape[0]*0.7):,:]
        ego_map[ego_map==VOID] = 0 # "ROAD"
        label_map[int(label_map.shape[0]*0.7):,:] = ego_map
        ego_map = pan_map[int(pan_map.shape[0]*0.7):,:]
        ego_map[ego_map==VOID] = 0 # "ROAD"
        pan_map[int(pan_map.shape[0]*0.7):,:] = ego_map

        return pan_map.astype(np.uint32), label_map.astype(np.uint8)

    def panoptic_single_core(self, proc_id, sem_files, inst_files):
        print('gothere')
        for sem_file, inst_file in zip(sem_files, inst_files):
            pan_map, label_map = self.sem_inst2pan(
                    sem_file, inst_file)
            Image.fromarray(pan_map).save(
                osp.join(PANOPTIC_DIR, sem_file.split('/')[-1]))
            Image.fromarray(label_map).save(
                osp.join(LABELMAP_DIR, sem_file.split('/')[-1]))

    def panoptic_multi_core(self, sem_files, inst_files):

        cpu_num = multiprocessing.cpu_count()//2
        sem_split = np.array_split(list(sem_files), cpu_num)
        inst_split = np.array_split(list(inst_files), cpu_num)
        assert(len(sem_split) == len(inst_split))
        print("Number of cores: %d, images per core: %d"%
            (cpu_num, len(sem_split[0])))
        workers = multiprocessing.Pool(processes=cpu_num)
        for proc_id, (sem_part, inst_part) in enumerate(zip(sem_split, inst_split)):
            # dbg = self.panoptic_single_core(proc_id, sem_part, inst_part)
            # import pdb; pdb.set_trace()
            workers.apply_async(self.panoptic_single_core,
                    (proc_id, sem_part, inst_part))
        workers.close()
        workers.join()

    def create_panoptic(self):
        # panoptic video annotations
        start_time = time.time()
        print('==> %s/labelmap/, %s/panoptic_inst/ ...'%(MODE, MODE))  
        # merge semantic segmentation and instance id map into panoptic format
        sem_files = [osp.join(SEMANTIC_DIR,x) for x in os.listdir(SEMANTIC_DIR) if '.png' in x]
        sem_files.sort()
        inst_files = [osp.join(INSTANCE_DIR,x) for x in os.listdir(INSTANCE_DIR) if '.png' in x]
        inst_files.sort()
        if not (len(sem_files) == len(inst_files)):
            raise ValueError('len semfiles != len inst_files')
        self.panoptic_multi_core(sem_files, inst_files) 
        print('Time taken by panoptic creation:', time.time()-start_time)



def create_json(id_converter, coco_output):
    # jsons file for instance masks
    start_time = time.time()
    print('==> %s/*.json'%(MODE))
    png_files = [x for x in os.listdir(SEMANTIC_DIR) if '.png' in x]
    png_files.sort()
    # rle=True by default
    image_info, annotation_info = create_json_multi_core(
            png_files, '', id_converter, neighbor=None, rle=True)
    pdb.set_trace()
    coco_output['images'] = image_info
    coco_output['annotations'] = annotation_info
    # let's assign ann_id to every annotations
    for idx, ann in enumerate(coco_output['annotations']):
        ann['id'] = idx
    # Always RLE = True
    save_name = 'instatnces_%s_st%02d_cityscapes_rle.json'%(MODE, args.stride)
    with open(osp.join(ROOT_DIR, save_name), 'w') as f:
        json.dump(coco_output, f)
    print('Time taken by json file creation:', time.time() - start_time)




if __name__=='__main__':
    import pdb

    id_converter = {x['color'][0]+x['color'][1]*256+x['color'][2]*256*256:x['id'] for x in CATEGORIES}
    ori2fcn = {x['ori_id'] : x['id'] for x in CATEGORIES}
    # panoptic video annotations
    panoptic_creator = panoptic_creator(id_converter, ori2fcn)
    panoptic_creator.create_panoptic()

    cat_things = [x for x in CATEGORIES if x['isthing']==1]
    coco_output = {
        'info': INFO,
        'licenses': LICENSES,
        'categories': cat_things,
        'images': [],
        'annotations': []
    }
    create_json(id_converter, coco_output)