import argparse
import os
import os.path as osp
import shutil
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
#### upsnet libraries
from upsnet.config.config import config, update_config
from upsnet.config.parse_args import parse_args
from lib.utils.logging import create_logger
from lib.utils.timer import Timer
# args = parse_args()
# logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
from upsnet.dataset import *
from upsnet.models import *
from upsnet.bbox.bbox_transform import bbox_transform, clip_boxes, expand_boxes
from lib.utils.callback import Speedometer
from lib.utils.data_parallel import DataParallel
from pycocotools.mask import encode as mask_encode
import pickle

import pdb


def im_detect(output_all, data, im_infos):

    scores_all = []
    pred_boxes_all = []
    pred_masks_all = []
    pred_ssegs_all = []
    pred_panos_all = []
    pred_pano_cls_inds_all = []
    cls_inds_all = []

    if len(data) == 1:
        output_all = [output_all]

    output_all = [{k: v.data.cpu().numpy() for k, v in output.items()} for output in output_all]

    for i in range(len(data)):
        im_info = im_infos[i]
        scores_all.append(output_all[i]['cls_probs'])
        pred_boxes_all.append(output_all[i]['pred_boxes'][:, 1:] / im_info[2])
        cls_inds_all.append(output_all[i]['cls_inds'])

        if config.network.has_mask_head:
            pred_masks_all.append(output_all[i]['mask_probs'])
        if config.network.has_fcn_head:
            pred_ssegs_all.append(output_all[i]['fcn_outputs'])
        if config.network.has_panoptic_head:
            pred_panos_all.append(output_all[i]['panoptic_outputs'])
            pred_pano_cls_inds_all.append(output_all[i]['panoptic_cls_inds'])

    return {
        'scores': scores_all,
        'boxes': pred_boxes_all,
        'masks': pred_masks_all,
        'ssegs': pred_ssegs_all,
        'panos': pred_panos_all,
        'cls_inds': cls_inds_all,
        'pano_cls_inds': pred_pano_cls_inds_all,
    }


def im_post(boxes_all, masks_all, scores, pred_boxes, pred_masks, cls_inds, num_classes, im_info):

    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0

    M = config.network.mask_size

    scale = (M + 2.0) / M


    ref_boxes = expand_boxes(pred_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    for idx in range(1, num_classes):
        segms = []
        cls_boxes = np.hstack([pred_boxes[idx == cls_inds, :], scores.reshape(-1, 1)[idx == cls_inds]])
        cls_pred_masks = pred_masks[idx == cls_inds]
        cls_ref_boxes = ref_boxes[idx == cls_inds]
        for _ in range(cls_boxes.shape[0]):

            if pred_masks.shape[1] > 1:
                padded_mask[1:-1, 1:-1] = cls_pred_masks[_, idx, :, :]
            else:
                padded_mask[1:-1, 1:-1] = cls_pred_masks[_, 0, :, :]
            ref_box = cls_ref_boxes[_, :]

            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > 0.5, dtype=np.uint8)
            im_mask = np.zeros((im_info[0], im_info[1]), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_info[1])
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_info[0])

            im_mask[y_0:y_1, x_0:x_1] = mask[
                                        (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                                        (x_0 - ref_box[0]):(x_1 - ref_box[0])
                                        ]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_encode(
                np.array(im_mask[:, :, np.newaxis], order='F')
            )[0]
            rle['counts'] = rle['counts'].decode()
            segms.append(rle)

            mask_ind += 1

        cls_segms[idx] = segms
        boxes_all[idx].append(cls_boxes)
        masks_all[idx].append(segms)


def single_gpu_test(model, data_loader, show=False, save_dir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    
    # #### DEBUG
    # import matplotlib
    # matplotlib.use('TkAgg')
    # def normalize(img):
    #     return (img-img.min())/(img.max()-img.min())
    # for i in range(100):
    #     dbg = dataset.prepare_test_img(i)
    #     img = normalize(dbg['img'][0].data.numpy().transpose(1,2,0))
    #     ref_img = normalize(dbg['ref_img'][0].data.numpy().transpose(1,2,0))
    #     next_img = normalize(dbg['next_img'][0].data.numpy().transpose(1,2,0))
    #     plt.subplot(231),plt.imshow(img)
    #     plt.subplot(232),plt.imshow(ref_img)
    #     plt.subplot(233),plt.imshow(next_img)
    #     plt.show()
    #     pdb.set_trace()

    pano_results = {
        'all_names':[],
        'all_ssegs':[],
        'all_panos':[],
        'all_pano_cls_inds':[],
        'all_pano_obj_ids':[]
        }        
    prog_bar = mmcv.ProgressBar(len(dataset))
    # best = [11,29,31,43]
    for i, data in enumerate(data_loader):
        filename = data['img_meta'][0].data[0][0]['filename'].split('/')[-1]
        # vid = int(filename.split('_')[0])
        # if vid not in best:
        #     continue
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result[:2])

        if len(result)>=3:
            # panoptic results
            pano_results['all_ssegs'].append(result[2]['fcn_outputs'].data.cpu().numpy()[0].astype(np.uint8))
            pano_results['all_panos'].append(result[2]['panoptic_outputs'].data.cpu().numpy()[0].astype(np.uint8))
            pano_results['all_pano_cls_inds'].append(result[2]['panoptic_cls_inds'].data.cpu().numpy())
            pano_results['all_names'].append(filename)
            if 'panoptic_det_obj_ids' in result[2]:
                pano_results['all_pano_obj_ids'].append(result[2]['panoptic_det_obj_ids'].data.cpu().numpy())

        if show:
            model.module.show_result(data, result, 
                        out_file=osp.join(save_dir, '%05d.png'%(i)))

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results, pano_results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)
    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--track', action='store_true')
    parser.add_argument('--json_out',help='output result file name without extension', type=str)
    parser.add_argument('--eval', type=str, nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints',
        'semantic_segm'],  help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--n_video', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='Cityscapes')
    parser.add_argument('--name', type=str, default='demo3')
    parser.add_argument('--txt_dir', type=str, default='demo3')
    parser.add_argument('--cfg', type=str, 
        default='upsnet/experiments/upsnet_resnet50_cityscapes_1gpu.yaml')
    
    args, rest = parser.parse_known_args()
    #### update config
    update_config(args.cfg)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    # print(config)
    # create models

    args = parse_args()
    gpus = [int(_) for _ in config.gpus.split(',')]
    # test_model = eval(config.symbol)().cuda(device=gpus[0])
    # pdb.set_trace()

    #### Original Content
    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    # dbg = dataset.prepare_test_img(12)
    # pdb.set_trace()
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
    
    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if args.load:
        outputs = mmcv.load(args.out)
        pano_pkl = args.out.replace('.pkl','_pano.pkl')
        outputs_pano = pickle.load(open(pano_pkl, 'rb'))
    else:
        if not distributed:
            model = MMDataParallel(model, device_ids=[0])
            outputs, outputs_pano = single_gpu_test(model, data_loader, args.show, save_dir=args.out.split('.pkl')[0])
        else:
            raise NotImplementedError
            model = MMDistributedDataParallel(model.cuda())
            outputs = multi_gpu_test(model, data_loader, args.tmpdir)

        with open(args.out, 'wb') as f:
            pickle.dump(outputs, f, protocol=2)

        with open(os.path.join(args.out.replace('.pkl','_pano.pkl')), 'wb') as f:
            pickle.dump(outputs_pano, f, protocol=2)

    # helper dataset from upsnet
    # config.dataset.dataset --> args.dataset
    if args.dataset == 'Viper':
        config.dataset.num_seg_classes = 23
        config.dataset.num_classes = 11
        config.dataset.name = args.name
    # elif args.dataset == "CityscapesExt":
    #     config.dataset.name = args.name
    eval_helper_dataset = eval(args.dataset)(image_sets=config.dataset.test_image_set.split('+'), flip=False, 
    result_path=args.out.split('.pkl')[0], phase='test')

    # *******************************************
    # EVAL: SEMANTIC SEGMENTATION
    # *******************************************
    # eval_helper_dataset.evaluate_ssegs(outputs_pano['all_ssegs'], args.out.replace('.pkl','_ssegs'), outputs_pano['all_names'])
    
    # *******************************************    
    # EVAL: PANOPTIC SEGMENTATION
    # *******************************************
    if args.dataset in ["Viper", "CityscapesExt"]:
        obj_ids = outputs_pano['all_pano_obj_ids'] if args.track else None
        pred_pans_2ch_ = eval_helper_dataset.get_unified_pan_result(outputs_pano['all_ssegs'],outputs_pano['all_panos'], outputs_pano['all_pano_cls_inds'], obj_ids=obj_ids, stuff_area_limit=config.test.panoptic_stuff_area_limit, names=outputs_pano['all_names'])
    else:
        pred_pans_2ch_ = eval_helper_dataset.get_unified_pan_result(outputs_pano['all_ssegs'],outputs_pano['all_panos'], outputs_pano['all_pano_cls_inds'], stuff_area_limit=config.test.panoptic_stuff_area_limit, names=outputs_pano['all_names'])

    pred_keys = [_ for _ in pred_pans_2ch_.keys()]
    pred_keys.sort()
    pred_pans_2ch = [pred_pans_2ch_[k] for k in pred_keys]
    del pred_pans_2ch_
    if not os.path.exists(os.path.join(args.out.split('.pkl')[0],args.txt_dir)):
        os.makedirs(os.path.join(args.out.split('.pkl')[0],args.txt_dir))
    if args.dataset in["Viper", "CityscapesExt"]:
        eval_helper_dataset.evaluate_panoptic(pred_pans_2ch, args.out.replace('.pkl','_pans_unified'), is_track=args.track, n_video=args.n_video, save_name=os.path.join(args.out.split('.pkl')[0],args.txt_dir))
    else:
        eval_helper_dataset.evaluate_panoptic(pred_pans_2ch, args.out.replace('.pkl','_pans_unified'))

    # *******************************************    
    # EVAL: BBOX & MASK
    # *******************************************
    # rank, _ = get_dist_info()
    # if args.out and rank == 0:
    #     print('\nwriting results to {}'.format(args.out))
    #     mmcv.dump(outputs, args.out)
    #     eval_types = args.eval
    #     if eval_types:
    #         print('Starting evaluate {}'.format(' and '.join(eval_types)))
    #         if eval_types == ['proposal_fast']:
    #             result_file = args.out
    #             coco_eval(result_file, eval_types, dataset.coco)
    #         else:
    #             if not isinstance(outputs[0], dict):
    #                 result_files = results2json(dataset, outputs, args.out, eval_types)
    #                 # we evaluate semantic_segm or panoptic segm offline
    #                 if 'semantic_segm' in eval_types:
    #                     print("semantic segmentation results are saved in {}.{}.json".format(args.out, 'semantic_segm'))
    #                     print("instance segmentation results are saved in {}.{}.json".format(args.out, 'segm'))
    #                     print("Please use COCO 2018 Panoptic Segmentation Task API(https://github.com/cocodataset/panopticapi) to combine instance segmentation and semantic segmentation and then evaluate Panoptic Segmentation results")
    #                     del eval_types[2]
    #                 if 'semantic_segm_all' in eval_types:
    #                     del eval_types[-1]

    #                 coco_eval(result_files, eval_types, dataset.coco)
    #             else:
    #                 for name in outputs[0]:
    #                     print('\nEvaluating {}'.format(name))
    #                     outputs_ = [out[name] for out in outputs]
    #                     result_file = args.out + '.{}'.format(name)
    #                     result_files = results2json(dataset, outputs_,
    #                                                 result_file)
    #                     coco_eval(result_files, eval_types, dataset.coco)

    # # Save predictions in the COCO json format
    # if args.json_out and rank == 0:
    #     if not isinstance(outputs[0], dict):
    #         results2json(dataset, outputs, args.json_out)
    #     else:
    #         for name in outputs[0]:
    #             outputs_ = [out[name] for out in outputs]
    #             result_file = args.json_out + '.{}'.format(name)
    #             results2json(dataset, outputs_, result_file)


if __name__ == '__main__':
    main()
