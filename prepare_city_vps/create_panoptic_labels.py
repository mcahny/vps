import datetime
import json
import os, sys
import os.path as osp
# import re
# import fnmatch
from PIL import Image
import numpy as np
sys.path.append('../')
from pycococreatortools import pycococreatortools
from pycocotools import mask
# import matplotlib.pyplot as plt

from city_default import CATEGORIES, INFO, LICENSES
# from merge_json import merge_json_files
import argparse
import shutil
import gc
import time
import multiprocessing
# import matplotlib.pyplot as plt
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='val', help='train/val/test')
parser.add_argument('--root_dir', type=str, default='data/city_ext/', help='root directory')
# parser.add_argument('--stride', type=int, default=1)
# parser.add_argument('--chunk', type=int, default=20)
# parser.add_argument('--panoptic', action='store_true')
# parser.add_argument('--jsons', action='store_true')
# parser.add_argument('--merge', action='store_true')
# parser.add_argument('--neighbor', action='store_true')
# parser.add_argument('--rle', action='store_true')
args = parser.parse_args()

MODE = args.mode
ROOT_DIR = args.root_dir
SEMANTIC_DIR = os.path.join(ROOT_DIR, MODE, 'cls')
INSTANCE_DIR = os.path.join(ROOT_DIR, MODE, 'inst')
LABELMAP_DIR = os.path.join(ROOT_DIR, MODE, 'labelmap')
os.makedirs(LABELMAP_DIR, exist_ok=True)
PANOPTIC_DIR = os.path.join(ROOT_DIR, MODE, 'panoptic_inst')
os.makedirs(PANOPTIC_DIR, exist_ok=True)
# BINARY_DIR = os.path.join(ROOT_DIR, 'binary_'+MODE)
# os.makedirs(LABELMAP_DIR, exist_ok=True)
# JSON_DIR = os.path.join(ROOT_DIR, 'jsons_%02d_'%(args.stride)+MODE)
# if not os.path.exists(JSON_DIR):
#     os.makedirs(JSON_DIR)
# JSON_OUT = '/home/code-base/data/viper/instances_viper_%s_coco.json'%(MODE)

####
VOID = 255

# def filter_for_png(root, files):
#     file_types = ['*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     files = [os.path.join(root, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
#     panoptic
#     return files


# def filter_for_annotations(root, v_dir, i_dir, files):
#     file_types = ['*.png']
#     file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
#     files = [os.path.join(root, v_dir, i_dir, f) for f in files]
#     files = [f for f in files if re.match(file_types, f)]
    
#     return files


# def rgb2int(arr):
#     """
#     Convert (N,...M,3)-array of dtype uint8 to a (N,...,M)-array of dtype int32
#     """
#     return arr[...,0]*(256**2)+arr[...,1]*256+arr[...,2]

# def rgb2vals(color, color2ind):
#     int_colors = rgb2int(color)
#     color2ind_keys = [k for k,v in color2ind.items()]
#     color2ind_values = [v for k,v in color2ind.items()]
#     int_keys = rgb2int(np.array(color2ind_keys, dtype='uint8'))
#     int_array = np.r_[int_colors.ravel(), int_keys]
#     uniq, index = np.unique(int_array, return_inverse=True)
#     color_labels = index[:int_colors.size]
#     key_labels = index[-len(color2ind):]

#     colormap = np.empty_like(int_keys, dtype='uint32')
#     colormap[key_labels] = color2ind_values
#     out = colormap[color_labels].reshape(color.shape[:2])
#     return out
    


# def semantic2label(sem_file, id_converter):
#     # read semantic mask
#     sem_map = np.array(Image.open(sem_file))
#     label_map = np.zeros((sem_map.shape[0], sem_map.shape[1]))
#     sem_ids = np.unique(sem_map)
    
#     for sem_id in sem_ids:
#         if sem_id not in id_converter:
#             continue
#         id = id_converter[sem_id]
#         mask = sem_map == sem_id
#         label_map[mask] = id
    
#     return label_map.astype(np.uint8)


#     valid_insts = []
#     for pan_id in np.unique(pan_map):
#         valid_inst = {}
        
#         cls_id = pan_id // OFFSET
#         inst_id = pan_id % OFFSET
#         # if background, pass
#         if inst_id == 0:
#             continue
#         binary_map = (pan_map == pan_id)
#         num_pixels = np.sum(binary_map)
#         #### IF EXTREMELY SMALL OBJECT, PASS
#         if num_pixels < SIZE_THR:
#             continue
#         # cls_id should be in id_converter dict.
#         if cls_id not in id_converter:
#             continue
#         fcn_id = id_converter[cls_id]
#         # Filter out stuff classes
#         if not ID2CATINFO[fcn_id]['isthing']:
#             continue
#         # # Filter out EGO-CENTRIC CAR
#         if ID2CATINFO[fcn_id]['name'] == 'car' and num_pixels > 10000:
#             ioa = np.sum(np.logical_and(ego_map, binary_map))/np.sum(ego_map)
#             if ioa > 0.5:
#                 continue

#         save_dir = os.path.join(BINARY_DIR, png_file.split('/')[-1].split('.')[0])
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)


# H, W = 1024, 2048


# def create_json_single_core(proc_id, png_files, inst_v_dir, id_converter, 
#                             neighbor=False, rle=False):
    
#     image_info = []
#     annotation_info = []
#     for idx, png_file in enumerate(png_files):
#         # iid = int(png_file.split('_')[0]+png_file.split('_')[3])
#         vid = int(png_file[:4])
#         iid = int(png_file[:9])
#         # print(iid)
#         # file_path = osp.join(INSTANCE_DIR, inst_v_dir, png_file)
#         file_path = osp.join(PANOPTIC_DIR, inst_v_dir, png_file)
#         insts = png2insts(file_path, id_converter, rle)

#         for inst in insts:
#             inst['image_id']=iid
#             if inst['fcn_id'] <= 10: # cuz stuff_max = 10 
#                 print('WRONG FCN_ID NUMBER !!!!!!!!!!!!!!!!!!!!')
#                 continue
#             inst['category_id'] = inst['fcn_id']
#             # inst['category_id'] = FCNID2TRAINID[inst['fcn_id']]
#             inst['width']=W
#             inst['height']=H
#         img_name = os.path.join(inst_v_dir, png_file.replace('final_mask','newImg8bit').replace('gtFine_color','leftImg8bit'))
#         im_info = pycococreatortools.create_image_info(iid, MODE,
#                                                             img_name, (W, H))
#         image_info.append(im_info)
#         annotation_info.extend(insts)
    
#     return image_info, annotation_info

# def inst2ann(inst, iid, ann_id=1):
#     binary_mask = np.array(Image.open(inst['bin_file']).convert('1')).astype(np.uint8)
#     annotation_info = pycococreatortools.create_instance_info(inst, binary_mask, iid, ann_id, tolerance=2, bounding_box=None)
    
#     return annotation_info

# OFFSET = 1000
# SIZE_THR = 8**2
ID2CATINFO = {x['id']:x for x in CATEGORIES}
# def png2insts(png_file, id_converter, rle=False):
#     pan_map = np.array(Image.open(png_file), dtype=np.uint32)
#     valid_insts = []
#     for pan_id in np.unique(pan_map):
#         fcn_id = pan_id // OFFSET
#         # inst_id = pan_id % OFFSET

#         if fcn_id == 0: # stuff classes
#             continue
#         obj_mask = pan_map == pan_id
#         area = np.sum(obj_mask)
        
#         ann_info = {
#             'fcn_id': int(fcn_id),
#             # 'inst_id': int(inst_id),
#             'inst_id' : int(pan_id),
#             'area': int(area),
#             'iscrowd': 0,
#         }
#         binary_map_encoded = \
#             mask.encode(np.asfortranarray(obj_mask.astype(np.uint8)))
#         bbox = mask.toBbox(binary_map_encoded)
#         if rle:
#             segmentation = pycococreatortools.binary_mask_to_rle(obj_mask)
#         else:
#             segmentation = pycococreatortools.binary_mask_to_polygon(obj_mask, tolerance=2)
#         if segmentation is None or len(segmentation) == 0:
#             print('==================== segmentation is NONE ! ')
#             continue
#         ann_info['bbox'] = bbox.tolist()
#         ann_info['segmentation'] = segmentation

#         valid_insts.append(ann_info)

#     return valid_insts


# def create_json_multi_core(png_files, inst_v_dir, id_converter, neighbor=False, rle=False):
    
#     cpu_num = multiprocessing.cpu_count()//2
#     png_files_split = np.array_split(list(png_files), cpu_num)
#     print("Number of cores: %d, images per core: %d"%
#         (cpu_num, len(png_files_split[0])))
#     workers = multiprocessing.Pool(processes=cpu_num)
#     processes = []

#     for proc_id, png_files_part in enumerate(png_files_split):
#         p = workers.apply_async(create_json_single_core,
#                                 (proc_id, png_files_part, inst_v_dir,
#                                     id_converter, neighbor, rle))
#         # p = create_json_single_core(proc_id, png_files_part, inst_v_dir, id_converter)
#         processes.append(p)
#     image_info = []
#     annotation_info = []
#     for p in processes:
#         outs = p.get()
#         image_info.extend(outs[0])
#         annotation_info.extend(outs[1])
#     workers.close()
#     workers.join()
#     return image_info, annotation_info


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
        # fcn_id = ori2fcn[inst_id // 1000]
        # # they must be same
        # if fcn_id_ != fcn_id and np.sum(obj_mask) > 1000:
        #     print('====== FCN_ID CORRECTION:', fcn_id_'==>',fcn_id_)
        #     pdb.set_trace()
        #     plt.imshow(obj_mask)
        #     plt.show()
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

    # import matplotlib.pyplot as plt
    # plt.subplot(121),plt.imshow(pan_map.astype(np.uint32))
    # plt.subplot(122),plt.imshow(label_map.astype(np.uint32))
    # plt.show()

    return pan_map.astype(np.uint32), label_map.astype(np.uint8)

def panoptic_multi_core(sem_files, inst_files, id_converter, oridfcn):
    cpu_num = multiprocessing.cpu_count()//2
    sem_split = np.array_split(list(sem_files), cpu_num)
    inst_split = np.array_split(list(inst_files), cpu_num)
    assert(len(sem_split) == len(inst_split))
    print("Number of cores: %d, images per core: %d"%
        (cpu_num, len(sem_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []

    for proc_id, (sem_part, inst_part) in enumerate(zip(sem_split, inst_split)):
        # dbg = panoptic_single_core(proc_id, sem_part, inst_part, id_converter, ori2fcn)
        # pdb.set_trace()
        workers.apply_async(panoptic_single_core,
                          (proc_id, sem_part, inst_part, id_converter, ori2fcn))
    workers.close()
    workers.join()


# FCNID2TRAINID = {x['id']:x['trainid'] for x in CATEGORIES}
if __name__=='__main__':
    
    # cat_things = [x for x in CATEGORIES if x['isthing']==1]
    # coco_output = {
    #     'info': INFO,
    #     'licenses': LICENSES,
    #     'categories': cat_things,
    #     'images': [],
    #     'annotations': []
    # }
    
    # id_converter = {x['ori_id']:x['id'] for x in CATEGORIES}
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
    # panoptic_single_core(12, sem_files, inst_files, id_converter)          
    # print('==> Total Time Elapsed:', time.time() - start_pano)
         

    # if args.jsons:
    #     args.mode='validation'
    #     nframes_per_video=30

    #     start_json = time.time()
    #     print('args.img_jsons --> IMG_JSONS_STR_MODE/*.JSONS')
    #     coco_output['images'] = []
    #     coco_output['annotations'] = []
    #     img_all_dir = '/data2/video_panoptic/data/cityscapes_ext/%s/img/'%(args.mode)
    #     png_files = [x for x in os.listdir(img_all_dir) if '.png' in x]
    #     print(len(png_files))
    #     png_files.sort()
    #     for idx, png_file in enumerate(png_files):
    #         vid = int(idx // nframes_per_video)+1
    #         fid = int(idx % nframes_per_video)+1
    #         iid = vid*10000+fid
    #         # img_name = os.path.join(img_all_dir, png_file)
    #         im_info = pycococreatortools.create_image_info(iid, MODE, png_file, (W, H))
    #         coco_output['images'].append(im_info)
    #         # coco_output['annotations'].append([])
    #     save_name = 'instances_val_01_city_im_frankfurt %s.json'%(args.mode.split('_')[-1])
    #     with open(os.path.join(ROOT_DIR, save_name),'w') as f:
    #         json.dump(coco_output, f)


                
     

