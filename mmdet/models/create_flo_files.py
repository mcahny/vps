#!/usr/bin/env python
### python lib
import os, sys, argparse, glob, re, math, copy, pickle
from datetime import datetime
import numpy as np
import os.path as osp
### torch lib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

### custom lib
from networks.resample2d_package.resample2d import Resample2d
# import networks
import models, losses, datasets
#import datasets
# import utils
# from utils.flow_utils import load_flow, vis_flow, save_flow
from utils import flow_utils, tools
from imageio import imsave
from PIL import Image
import json
import mmcv

import pdb

class Object(object):
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # arguments
    args = Object()
    args.rgb_max = 255.0
    args.fp16 = False
    args.data_root = '/data2/video_panoptic/data/cityscapes/'
    args.ref_root = '/data3/leftImg8bit_sequence/'
    args.ann_dir = osp.join(args.data_root,'annotations')
    args.set_name = 'val'
    args.flow_folder = osp.join(args.data_root,'flows')
    # loading flownet2 
    FlowNet2 = models.FlowNet2(args, requires_grad=False)
    model_filename = os.path.join("./", "FlowNet2_checkpoint.pth.tar")
    print("===> Load: %s" %model_filename)
    checkpoint = torch.load(model_filename)
    FlowNet2.load_state_dict(checkpoint['state_dict'])
    FlowNet2 = FlowNet2.cuda()
    # sub modules 
    flow_warping = Resample2d().cuda()
    downsampler = nn.AvgPool2d((2, 2), stride=2).cuda()
    
    # read image pairs
    with open(osp.join(args.ann_dir,
        'instancesonly_pano_gtFine_%s.json'%(args.set_name)),'rb') as f:
        ann = json.load(f)

    file_list = [x['file_name'] for x in ann['images']]
    for idx, file_name in enumerate(file_list):
        if not idx%10:
            print(idx,'/',len(file_list))
        fid = int(file_name.split('_')[-2])
        img1_path = osp.join(args.data_root, args.set_name, file_name)
        city_name = file_name.split('_')[0]
        if args.set_name == 'train':
            img2_path = osp.join(args.data_root, args.set_name, 
                file_name.replace('%06d_leftImg8bit'%(fid), '%06d_leftImg8bit'%(fid-1)))
        else:
            img2_path = osp.join(args.ref_root, args.set_name, city_name,
                file_name.replace('%06d_leftImg8bit'%(fid), '%06d_leftImg8bit'%(fid-1)))

        img1 = mmcv.imread(img1_path)
        img2 = mmcv.imread(img2_path)
        H,W,_ = img1.shape
        images = [img1, img2]
        images = np.array(images).transpose(3,0,1,2) # C,2,H,W
        images = torch.from_numpy(images.astype(np.float32)).unsqueeze(0)
        images = images.cuda() # 1,C,2,H,W
        flow_pr = FlowNet2(images)
        flow_pr = flow_pr[0].data.cpu().numpy().transpose(1,2,0)
        flow_pr = mmcv.imresize(flow_pr, (W//2,H//2))
        flow_name = osp.join(args.flow_folder, 
            file_name.replace('leftImg8bit.png','%06d.flo'%(fid-1)))
        flow_utils.writeFlow(flow_name, flow_pr)
        # save visualization
        # flow_pr_ = flow_utils.readFlow(flow_name)
        # vis_pr = flow_utils.vis_flow(flow_pr_)
        # imsave(file_name.replace('leftImg8bit','%06d'%(fid-1)),vis_pr)
        # pdb.set_trace()

    print('Finished saving all .flo files...')

