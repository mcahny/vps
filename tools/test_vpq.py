import argparse
import os
import os.path as osp
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
# from mmdet.apis import init_dist
from mmdet.core import wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
#### upsnet libraries
from tools.config.config import config, update_config
# from upsnet.config.parse_args import parse_args
# from lib.utils.logging import create_logger
# from lib.utils.timer import Timer
# args = parse_args()
# logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
from tools.dataset import *
#from upsnet.models import *
#from upsnet.bbox.bbox_transform import bbox_transform, clip_boxes, expand_boxes
# from lib.utils.callback import Speedometer
# from lib.utils.data_parallel import DataParallel
# from pycocotools.mask import encode as mask_encode
import pickle
import pdb


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset

    pano_results = {
        'all_names':[],
        'all_ssegs':[],
        'all_panos':[],
        'all_pano_cls_inds':[],
        'all_pano_obj_ids':[]
        }        
    prog_bar = mmcv.ProgressBar(len(dataset))

    for i, data in enumerate(data_loader):
        filename = \
            data['img_meta'][0].data[0][0]['filename'].split('/')[-1]

        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        # mask results
        results.append(result[:2])

        # panoptic results
        if len(result)>=3:
            
            pano_results['all_ssegs'].append(
                result[2]['fcn_outputs'].data.cpu(
                    ).numpy()[0].astype(np.uint8))
            pano_results['all_panos'].append(
                result[2]['panoptic_outputs'].data.cpu(
                    ).numpy()[0].astype(np.uint8))
            pano_results['all_pano_cls_inds'].append(
                result[2]['panoptic_cls_inds'].data.cpu().numpy())
            pano_results['all_names'].append(filename)
            #### with Track head output: obj_ids w.r.t. reference frame
            if 'panoptic_det_obj_ids' in result[2]:
                pano_results['all_pano_obj_ids'].append(
                    result[2]['panoptic_det_obj_ids'].data.cpu().numpy())

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()

    return results, pano_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--gpus', type=str, default='0' )
    # parser.add_argument('--track', action='store_true')
    # parser.add_argument('--json_out',help='output result file name without extension', type=str)
    # parser.add_argument('--eval', type=str, nargs='+',
    #     choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints',
    #     'semantic_segm'],  help='eval types')
    # parser.add_argument('--show', action='store_true', help='show results')
    # parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    # parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    # parser.add_argument('--n_video', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='Cityscapes')
    parser.add_argument('--name', type=str, default='val0429')
    # parser.add_argument('--txt_dir', type=str, default='demo3')
    parser.add_argument('--test_config', type=str, 
        default='configs/cityscapes/test_cityscapes_1gpu.yaml')
    
    # ---- VPQ - specific arguments
    parser.add_argument('--has_track', action='store_true')
    parser.add_argument('--n_video', type=int, default=100)
    parser.add_argument('--txt_dir', type=str, default='val0429')

    parser.add_argument('--pan_gt_folder', type=str, 
                        default='data/cityscapes_ext/val/panoptic_video_vivid/')
    parser.add_argument('--pan_gt_json_file', type=str,
                        default='data/cityscapes_ext/cityscapes_ext_panoptic_val_video.json')
    # (pan_gt_json_file=None, pan_gt_folder=None)
    
    args, rest = parser.parse_known_args()
    #### update config
    update_config(args.test_config)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

# from easydict import EasyDict as edict

def main():

    args = parse_args()
    gpus = [int(_) for _ in args.gpus.split(',')]
    if args.out is not None and not args.out.endswith(('.pkl', 'pickle')):
        raise ValueError("The output file must be a .pkl file.")

    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backedns.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # launcher = None, distributed = False by default.
    distributed = False

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    # build the model and load checkpoint
    model = build_detector(cfg.model, 
                           train_cfg=None, 
                           test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, 
                                 args.checkpoint, 
                                 map_location='cpu')

    # E.g., Cityscapes has 8 things CLASSES.
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    # If _mask.pkl & _pano.pkl results are saved already, load = True.
    if args.load:
        outputs_mask = mmcv.load(args.out.replace('.pkl','_mask.pkl'))
        pano_pkl = args.out.replace('.pkl','_pano.pkl')
        outputs_pano = pickle.load(open(pano_pkl, 'rb'))
    else:
        # TODO: to support multi-gpu inference
        model = MMDataParallel(model, device_ids=[gpus[0]])
        # args.show = False
        outputs_mask, outputs_pano = \
                single_gpu_test(model,
                                data_loader)
        # save the outputs as .pkl files.
        with open(args.out.replace('.pkl','_mask.pkl'), 'wb') as f:
            pickle.dump(outputs_mask, f, protocol=2)
        with open(args.out.replace('.pkl','_pano.pkl'), 'wb') as f:
            pickle.dump(outputs_pano, f, protocol=2)

    # helper dataset from upsnet
    # config.dataset.dataset --> args.dataset
    # args.dataset = 'Cityscapes'
    # config.dataset.test_image_set = 'val'
    # result_path = 'val'
    # eval_helper_dataset = eval(args.dataset)(
    #         image_sets=config.dataset.test_image_set.split('+'), 
    #         # image_sets=args.out.split('.pkl')[0],
    #         flip=False,
    #         result_path=args.out.split('.pkl')[0], 
    #         phase='test')
    eval_helper_dataset = eval(args.dataset)()


    # # *******************************************
    # # EVAL: SEMANTIC SEGMENTATION
    # # *******************************************
    # print("==> Semantic Segmentation results will be saved at:")
    # print("---", args.out.split('.pkl')[0]+'_ssegs/')
    # eval_helper_dataset.evaluate_ssegs(
    #         outputs_pano['all_ssegs'], 
    #         args.out.replace('.pkl','_ssegs'), 
    #         outputs_pano['all_names'])

    # *******************************************    
    # EVAL: VIDEO PANOPTIC SEGMENTATION
    # *******************************************
    print("==> Video Panoptic Segmentation results will be saved at:")
    print("---", args.out.split('.pkl')[0]+'_pans_unified/')

    obj_ids = outputs_pano['all_pano_obj_ids']
    pred_pans_2ch_ = eval_helper_dataset.get_unified_pan_result(
            outputs_pano['all_ssegs'],
            outputs_pano['all_panos'], 
            outputs_pano['all_pano_cls_inds'], 
            obj_ids=obj_ids, 
            stuff_area_limit=config.test.panoptic_stuff_area_limit, 
            names=outputs_pano['all_names'])

    pred_keys = [_ for _ in pred_pans_2ch_.keys()]
    pred_keys.sort()
    # pred_keys = pred_keys[4::5]
    pred_pans_2ch = [pred_pans_2ch_[k] for k in pred_keys]
    del pred_pans_2ch_
    # ******************************
    # ['0005_0025_frankfurt_000000_001736_newImg8bit.png', '0005_0026_frankfurt_000000_001741_newImg8bit.png', '0005_0027_frankfurt_000000_001746_newImg8bit.png', '0005_0028_frankfurt_000000_001751_leftImg8bit.png']
    
    # ******************************
    # if not osp.exists(osp.join(args.out.split('.pkl')[0],args.txt_dir)):
    #     os.makedirs(osp.join(args.out.split('.pkl')[0],args.txt_dir))
    
    eval_helper_dataset.evaluate_panoptic_video(
            pred_pans_2ch, args.out.replace('.pkl','_pans_unified'),
            pan_gt_json_file=args.pan_gt_json_file,
            pan_gt_folder=args.pan_gt_folder,
            n_video=args.n_video,
            save_name=args.txt_dir)


if __name__ == '__main__':
    main()
