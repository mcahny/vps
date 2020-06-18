import os
import os.path as osp
import shutil
import argparse
import json

parser = argparse.ArgumentParser(description='Fetch City sequence images')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--src_dir', type=str, default='data/leftImg8bit_sequence/val')
parser.add_argument('--dst_dir', type=str, default='data/city_ext')
args = parser.parse_args()

mode = args.mode
src_dir = args.src_dir
dst_dir = osp.join(args.dst_dir, mode)

print('==> %s/img ...'%(mode))
os.makedirs(osp.join(dst_dir,'img'), exist_ok=True)
with open(osp.join(args.dst_dir, 'panoptic_im_%s_city_vps.json'%(mode)),'r') as f:
    im_json = json.load(f)
# Copy cls, inst files from the original Cityscapes
images = im_json['images']
for img in images:
    filename = img['file_name']
    name = img['id'][len('0000_0000_'):]
    prefix = img['id'][:len('0000_0000_')]
    city = name.split('_')[0]
    # Copy Cityscapes-sequence images
    srcfile = osp.join(src_dir, city, name+'_leftImg8bit.png')
    dstfile = osp.join(dst_dir, 'img', filename)
    shutil.copy(srcfile, dstfile)
print('Fetched Cityscapes images.')

if mode == 'val' or mode == 'test':
    print('==> %s/img_all ...'%(mode))
    os.makedirs(osp.join(dst_dir,'img_all'), exist_ok=True)
    with open(osp.join(args.dst_dir, 'im_all_info_%s_city_vps.json'%(mode)),'r') as f:
        im_json = json.load(f)
    images = im_json['images']
    for img in images:
        filename = img['file_name']
        city = filename.split('_')[0]
        srcfile = osp.join(src_dir, city, filename)
        dstfile = osp.join(dst_dir, 'img_all', filename)
        shutil.copy(srcfile, dstfile)
    print('Fetched Cityscapes-sequence images.')






