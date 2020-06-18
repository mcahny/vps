import os
import os.path as osp
import shutil
import argparse
import json

parser = argparse.ArgumentParser(description='Merge City + City-VPS datasets')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--src_dir', type=str, default='data/cityscapes/gtFine_trainvaltest/gtFine/val')
parser.add_argument('--dst_dir', type=str, default='data/city_ext')
args = parser.parse_args()

mode = args.mode
src_dir = args.src_dir
dst_dir = osp.join(args.dst_dir, mode)

with open(osp.join(args.dst_dir, 'panoptic_im_%s_city_vps.json'%(mode)),'r') as f:
    im_json = json.load(f)

print('==> %s/cls, %s/inst ...'%(mode, mode))
# Copy cls, inst files from the original Cityscapes
images = im_json['images']
for img in images:
    filename = img['file_name']
    # Copy 20th-frame annotations
    if "leftImg8bit" in filename:      
        name = img['id'][len('0000_0000_'):]
        prefix = img['id'][:len('0000_0000_')]
        city = name.split('_')[0]

        srcfile = osp.join(src_dir, city, name+'_gtFine_color.png')
        dstfile = osp.join(dst_dir, 'cls', prefix+name+'_gtFine_color.png')
        shutil.copy(srcfile, dstfile)

        srcfile = osp.join(src_dir, city, name+'_gtFine_instanceIds.png')
        dstfile = osp.join(dst_dir, 'inst', prefix+name+'_gtFine_instanceIds.png')
        shutil.copy(srcfile, dstfile)
    
print('Copied Cityscapes 20th-frame annotations.')




