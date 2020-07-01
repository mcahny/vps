import datetime
import json
import os, sys
import os.path as osp
from PIL import Image
import numpy as np
sys.path.append('../')
from pycococreatortools import pycococreatortools
from pycocotools import mask

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

    # Heuristics for the Ego-Centric Car
    ego_map = label_map[int(label_map.shape[0]*0.7):,:]
    ego_map[ego_map==VOID] = 0 # "ROAD"
    label_map[int(label_map.shape[0]*0.7):,:] = ego_map

    ego_map = pan_map[int(pan_map.shape[0]*0.7):,:]
    ego_map[ego_map==VOID] = 0 # "ROAD"
    pan_map[int(pan_map.shape[0]*0.7):,:] = ego_map

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


def create_panoptic(id_converter, ori2fcn):
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


# JSON FILE CREATION
OFFSET = 1000
SIZE_THR = 8**2
ID2CATINFO = {x['id']:x for x in CATEGORIES}
def png2insts(png_file, id_converter, rle=False):
    pan_map = np.array(Image.open(png_file), dtype=np.uint32)
    valid_insts = []
    for pan_id in np.unique(pan_map):
        fcn_id = pan_id // OFFSET
        # inst_id = pan_id % OFFSET
        if fcn_id == 0: # stuff classes
            continue
        obj_mask = pan_map == pan_id
        area = np.sum(obj_mask)
        
        ann_info = {
            'fcn_id': int(fcn_id),
            # 'inst_id': int(inst_id),
            'inst_id' : int(pan_id),
            'area': int(area),
            'iscrowd': 0,
        }
        binary_map_encoded = \
            mask.encode(np.asfortranarray(obj_mask.astype(np.uint8)))
        bbox = mask.toBbox(binary_map_encoded)
        if rle:
            segmentation = pycococreatortools.binary_mask_to_rle(obj_mask)
        else:
            segmentation = pycococreatortools.binary_mask_to_polygon(obj_mask, tolerance=2)
        if segmentation is None or len(segmentation) == 0:
            print('Warning: segmentation is NONE ! ')
            continue
        ann_info['bbox'] = bbox.tolist()
        ann_info['segmentation'] = segmentation

        valid_insts.append(ann_info)

    return valid_insts


H, W = 1024, 2048
def create_json_single_core(proc_id, png_files, inst_v_dir, id_converter, 
                            neighbor=False, rle=False):
    
    image_info = []
    annotation_info = []
    for idx, png_file in enumerate(png_files):
        # iid = int(png_file.split('_')[0]+png_file.split('_')[3])
        vid = int(png_file[:4])
        iid = int(png_file[:9])
        # print(iid)
        # file_path = osp.join(INSTANCE_DIR, inst_v_dir, png_file)
        file_path = osp.join(PANOPTIC_DIR, inst_v_dir, png_file)
        insts = png2insts(file_path, id_converter, rle)

        for inst in insts:
            inst['image_id']=iid
            if inst['fcn_id'] <= 10: # cuz stuff_max = 10 
                print('Warning: wrong FCN id')
                continue
            inst['category_id'] = inst['fcn_id']
            # inst['category_id'] = FCNID2TRAINID[inst['fcn_id']]
            inst['width']=W
            inst['height']=H
        img_name = os.path.join(inst_v_dir, png_file.replace('final_mask','newImg8bit').replace('gtFine_color','leftImg8bit'))
        im_info = pycococreatortools.create_image_info(
                iid, MODE, img_name, (W, H))
        image_info.append(im_info)
        annotation_info.extend(insts)
    
    return image_info, annotation_info


def create_json_multi_core(png_files, inst_v_dir, id_converter, 
                           neighbor=False, rle=False):
    
    cpu_num = multiprocessing.cpu_count()//2
    png_files_split = np.array_split(list(png_files), cpu_num)
    print("Number of cores: %d, images per core: %d"%
        (cpu_num, len(png_files_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, png_files_part in enumerate(png_files_split):
        p = workers.apply_async(create_json_single_core,
                                (proc_id, png_files_part, inst_v_dir,
                                    id_converter, neighbor, rle))

        processes.append(p)
    image_info = []
    annotation_info = []
    for p in processes:
        outs = p.get()
        image_info.extend(outs[0])
        annotation_info.extend(outs[1])
    workers.close()
    workers.join()
    return image_info, annotation_info

def create_json(id_converter, coco_output):
    # jsons file for instance masks
    start_time = time.time()
    print('==> %s/*.json'%(MODE))
    png_files = [x for x in os.listdir(SEMANTIC_DIR) if '.png' in x]
    png_files.sort()
    # rle=True by default
    image_info, annotation_info = create_json_multi_core(
            png_files, '', id_converter, neighbor=None, rle=True)
    coco_output['images'] = image_info
    coco_output['annotations'] = annotation_info
    # let's assign ann_id to every annotations
    for idx, ann in enumerate(coco_output['annotations']):
        ann['id'] = idx
    # Always RLE = True
    save_name = 'instances_%s_cityscapes_rle.json'%(MODE)
    with open(osp.join(ROOT_DIR, save_name), 'w') as f:
        json.dump(coco_output, f)
    print('Time taken by json file creation:', time.time() - start_time)




if __name__=='__main__':
    import pdb

    id_converter = {x['color'][0]+x['color'][1]*256+x['color'][2]*256*256:x['id'] for x in CATEGORIES}
    ori2fcn = {x['ori_id'] : x['id'] for x in CATEGORIES}
    # panoptic video annotations
    # create_panoptic(id_converter, ori2fcn)

    cat_things = [x for x in CATEGORIES if x['isthing']==1]
    coco_output = {
        'info': INFO,
        'licenses': LICENSES,
        'categories': cat_things,
        'images': [],
        'annotations': []
    }
    create_json(id_converter, coco_output)