import torch
import torch.nn.functional as F
import os
import os.path as osp
from mmdet.core import (bbox2result, bbox2roi, roi2bbox, 
                        build_assigner, build_sampler, 
                        bbox2result_with_id, bbox_overlaps)
from .two_stage import TwoStageDetector
from ..import builder
from ..registry import DETECTORS
# Borrowed from Xiong et al. "UPSNet"
from ..utils.unary_logits import (MaskTerm, SegTerm, 
                                  MaskMatching, MaskFcnTerm) 
from ..utils.mask_removal import MaskRemoval
from ..utils.mask_roi import MaskROI
from ..flow_modules import FlowNet2
from ..flow_modules.resample2d_package.resample2d import Resample2d
from ..utils.flow_utils import denormalize, rgb_denormalize, vis_flow
import numpy as np
import matplotlib.pyplot as plt
import mmcv
import pdb

@DETECTORS.register_module
class PanopticTrackFlowTcea(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 extra_neck=None,
                 panoptic=None,
                 track_head=None,
                 shared_head=None,
                 pretrained=None):
        super(PanopticTrackFlowTcea, self).__init__(
            backbone=backbone,
            neck=neck,
            extra_neck=extra_neck,
            panoptic=panoptic,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            track_head=track_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        
        if hasattr(self.train_cfg, 'class_mapping'):
            self.class_mapping = self.train_cfg['class_mapping']
        elif hasattr(self.test_cfg, 'class_mapping'):
            self.class_mapping = self.test_cfg['class_mapping']
        else:
            self.class_mapping = None
        # for panoptic head
        self.seg_term = SegTerm(
                self.panopticFPN.num_classes, 
                box_scale=1/4.0,
                class_mapping=self.class_mapping)
        self.mask_term = MaskTerm(
                self.panopticFPN.num_classes, 
                box_scale=1/4.0,
                class_mapping=self.class_mapping)
        self.mask_fcn_term = MaskFcnTerm(
                self.panopticFPN.num_classes, 
                box_scale=1/4.0,
                class_mapping=self.class_mapping)
        self.mask_matching = MaskMatching(
                self.panopticFPN.num_classes, 
                ignore_label=255,
                class_mapping=self.class_mapping)
        self.mask_roi_panoptic = MaskROI(
                clip_boxes=True, bbox_class_agnostic=False, top_n=100, 
                num_classes=self.panopticFPN.num_things_classes+1, 
                nms_thresh=0.5, class_agnostic=True, score_thresh=0.6)
        self.mask_removal = MaskRemoval(fraction_threshold=0.3)
        # flow module
        if (hasattr(self.train_cfg, 'flownet2') 
            or hasattr(self.test_cfg, 'flownet2')):
            self.mean=[123.675, 116.28, 103.53]
            self.std=[58.395, 57.12, 57.375]
            class Object(object):
                pass
            flow2_args = Object()
            flow2_args.rgb_max = 255.0
            flow2_args.fp16 = False
            self.flownet2 = FlowNet2(flow2_args, requires_grad=False)
            model_filename = osp.join(os.getcwd(), 'work_dirs',
                'flownet', 'FlowNet2_checkpoint.pth.tar')
            print("==> Inside PanopticTrackFlowTcea (Device ID: %d)"%(
                  torch.cuda.current_device()))
            print("--- Load flow module: %s"%model_filename)
            checkpoint = torch.load(model_filename)
            self.flownet2.load_state_dict(checkpoint['state_dict'])
            self.flownet2 = self.flownet2.cuda()
            self.flownet2.eval()
            self.flow_warping = Resample2d().cuda()
        
    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def compute_flow(self, img, ref_img, scale_factor=1):

        rgb = denormalize(img, self.std, self.mean)
        ref_rgb = denormalize(ref_img, self.std, self.mean)
        rgbs = torch.stack([rgb, ref_rgb], dim=2)
        H,W = rgbs.size(-2), rgbs.size(-1)
        #### Pad zeros
        # Need: size generalization
        if H == 800 and W == 1600:
            rgbs = F.pad(rgbs,(0,64,0,32))
        elif H == 200 and W == 400:
            rgbs = F.pad(rgbs,(0,48,0,56))
        # flownet input must be divisible by 64.
        assert rgbs.size(-2)%64==0 and rgbs.size(-1)%64==0
        self.flownet2.cuda(rgbs.device)
        self.flownet2.eval()
        flow = self.flownet2(rgbs)
        #### Trim zeros
        indH = torch.arange(0,H).cuda(rgbs.device)
        indW = torch.arange(0,W).cuda(rgbs.device)
        flow = torch.index_select(flow,-2,indH)
        flow = torch.index_select(flow,-1,indW)
        # flow_ = flow[:,:,:H,:W]
        if scale_factor != 1:
            flow = F.interpolate(flow, scale_factor=scale_factor, 
                    mode='bilinear', align_corners=False) * scale_factor
        return flow, None


    def forward_train(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      gt_semantic_seg_Nx=None,
                      proposals=None,
                      ref_img=None, # images of reference frame
                      ref_bboxes=None, # gt bbox of reference frame
                      ref_labels=None,
                      ref_masks=None,
                      ref_semantic_seg=None,
                      ref_semantic_seg_Nx=None,
                      ref_obj_ids=None,
                      gt_pids=None, # gt ids of target objs mapped to reference objs
                      gt_obj_ids=None,
                      gt_flow=None,                     
                      ):
        # #### DEBUG
        # import matplotlib.pyplot as plt
        # flow_warping = Resample2d().cuda()
        # flow, _ = self.compute_flow(img.clone().detach(), ref_img.clone().detach(), 0.25)
        # flo = flow.data[0].cpu().numpy().transpose(1,2,0)
        # vis = vis_flow(flo)
        # plt.subplot(131),plt.imshow(rgb_denormalize(img.data[0].cpu().numpy().transpose(1,2,0),self.std, self.mean).astype(np.uint8))
        # plt.subplot(132),plt.imshow(vis)
        # flow, _ = self.compute_flow(img.clone().detach(), ref_img.clone().detach())
        # flo2 = flow.data[0].cpu().numpy().transpose(1,2,0)
        # vis2 = vis_flow(flo2)
        # plt.subplot(133),plt.imshow(vis2)
        # plt.show()

        # warp_img = flow_warping(ref_img, flow)
        # diff_after = abs(img-warp_img)
        # diff_before = abs(img-ref_img)
        
        # img_ = rgb_denormalize(img.data[0].cpu().numpy().transpose(1,2,0),self.std, self.mean).astype(np.uint8)
        # ref_img_ = rgb_denormalize(ref_img.data[0].cpu().numpy().transpose(1,2,0),self.std, self.mean).astype(np.uint8)
        # warp_img_ = rgb_denormalize(warp_img.data[0].cpu().numpy().transpose(1,2,0),self.std, self.mean).astype(np.uint8)
        # diff_before_ = rgb_denormalize(diff_before.data[0].cpu().numpy().transpose(1,2,0),self.std, self.mean).astype(np.uint8)
        # diff_after_ = rgb_denormalize(diff_after.data[0].cpu().numpy().transpose(1,2,0),self.std, self.mean).astype(np.uint8)
        # print(gt_obj_ids)
        # print(ref_obj_ids)
        # print(gt_pids)
        # plt.subplot(251),plt.imshow(img_)
        # plt.subplot(252),plt.imshow(ref_img_)
        # mask = np.zeros_like(img_[:,:,0])
        # gt_obj_ids = gt_obj_ids[0].data.cpu().numpy()
        # gt_masks = gt_masks[0]
        # for i in range(len(gt_masks)):
        #     oid = gt_obj_ids[i]
        #     m = gt_masks[i]
        #     mask[m==1] = oid+1
        # ref_mask = np.zeros_like(img_[:,:,0])
        # ref_obj_ids = ref_obj_ids[0].data.cpu().numpy()
        # ref_masks = ref_masks[0]
        # for i in range(len(ref_masks)):
        #     oid = ref_obj_ids[i]
        #     m = ref_masks[i]
        #     ref_mask[m==1] = oid+1
        # plt.subplot(253),plt.imshow(mask)
        # plt.subplot(254),plt.imshow(ref_mask)
        # plt.subplot(255),plt.imshow(vis)
        # plt.subplot(256),plt.imshow(warp_img_)
        # plt.subplot(257),plt.imshow(diff_before_)
        # plt.subplot(258),plt.imshow(diff_after_)
        # gt_fcn = gt_semantic_seg.data[0].cpu().numpy()[0]
        # ref_fcn = ref_semantic_seg.data[0].cpu().numpy()[0]
        # plt.subplot(259),plt.imshow(gt_fcn)
        # plt.subplot(2,5,10),plt.imshow(ref_fcn)
        # plt.show()
        # pdb.set_trace()   

        losses = dict()

        # ********************************
        # Initial Flow and Feature Warping
        # ******************************** 
        flowR2T, _ = self.compute_flow(img.clone(), ref_img.clone(), scale_factor=0.25)
        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_img)
        x = self.extra_neck(x, ref_x, flowR2T)

        # **********************************
        # FCN Semantic Head forward and loss
        # **********************************
        if hasattr(self, 'panopticFPN') and self.panopticFPN is not None:
            # semantic FCN GT
            gt_semantic_seg = gt_semantic_seg.long()
            gt_semantic_seg = gt_semantic_seg.squeeze(1)
            fcn_output, fcn_score = \
                    self.panopticFPN(x[0:self.panopticFPN.num_levels])
            loss_fcn = F.cross_entropy(
                    fcn_output, gt_semantic_seg, ignore_index=255)
            loss_fcn = {'loss_segm': loss_fcn}
            losses.update(loss_fcn)

        # ***************************
        # RPN forward and loss
        # ***************************
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                    *rpn_loss_inputs, 
                    gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals
        
        # *******************************
        # assign gts and sample proposals
        # *******************************
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                    self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(
                        proposal_list[i], gt_bboxes[i], 
                        gt_bboxes_ignore[i], gt_labels[i], gt_pids[i])
                sampling_result = bbox_sampler.sample(
                        assign_result,
                        proposal_list[i],
                        gt_bboxes[i],
                        gt_labels[i],
                        gt_pids[i],
                        feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # *******************************
        # bbox head forward and loss
        # *******************************
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_img_n = [res.bboxes.size(0) for res in sampling_results]
            ref_rois = bbox2roi(ref_bboxes)
            ref_bbox_img_n = [x.size(0) for x in ref_bboxes]
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                    x[:self.bbox_roi_extractor.num_inputs], rois)
            ref_bbox_feats = self.bbox_roi_extractor(
                    ref_x[:self.bbox_roi_extractor.num_inputs], ref_rois)
            # if self.with_shared_head:
            #     bbox_feats = self.shared_head(bbox_feats)

            cls_score, bbox_pred = self.bbox_head(bbox_feats)
            # fetch bbox and object_id targets
            bbox_targets, (ids, id_weights) = \
                    self.bbox_head.get_id_target(
                            sampling_results, gt_bboxes, 
                            gt_labels, self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(
                    cls_score, bbox_pred, *bbox_targets)
            losses.update(loss_bbox)

            # ******************************* 
            # Track Head forward and loss
            # ******************************* 
            if self.with_track:
                # bbox_feats 1B:[128,256,7,7], 
                # ref_bbox_feats 1B:[n,256,7,7]
                match_score = self.track_head(
                        bbox_feats, ref_bbox_feats,
                        bbox_img_n, ref_bbox_img_n)
                # ids: where is it from?
                # match_score:[128,n+1], ids:[128], id_weights:[128]
                loss_match = self.track_head.loss(
                        match_score, ids, id_weights)
                losses.update(loss_match)
            
        # *******************************    
        # mask head forward and loss
        # *******************************
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)

                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)
            mask_targets = self.mask_head.get_target(
                    sampling_results, gt_masks, self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(
                    mask_pred, mask_targets, pos_labels)
            losses.update(loss_mask)

        # ***************************************
        # PANOPTIC HEAD - Only for BATCH SIZE: 1
        # ***************************************
        if hasattr(self.train_cfg, 'loss_pano_weight'):
            # extract gt rois for panpotic head
            gt_rois = bbox2roi(gt_bboxes) # [#bbox, 5]
            cls_idx = gt_labels[0] # [#bbox] / batch_size must be 1
            # fcn_score # [1,20,200,400]
            # compute mask logits with gt rois
            mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], gt_rois)
                    # [#bbox,256,14,14]
            mask_score = self.mask_head(mask_feats) # [#bbox,#things+1,28,28], #things+1=9
            nobj,_,H,W = mask_score.shape
            mask_score = mask_score.gather(
                    1, cls_idx.view(-1,1,1,1).expand(-1,-1,H,W)) 
                                # [#bbox,1,28,28]
            # compute panoptic logits
            seg_stuff_logits, seg_inst_logits = self.seg_term(
                    cls_idx, fcn_score, gt_rois)
            mask_logits = self.mask_term(
                    mask_score, gt_rois, cls_idx, fcn_score)
            # panoptic_logits: [1,#stuff+#bbox,200,400]
            panoptic_logits = torch.cat(
                    [seg_stuff_logits, (seg_inst_logits + mask_logits)],
                    dim=1)
            # generate gt for panoptic head
            # added for panoptic gt generation : gt_masks_4x
            gt_masks_4x = gt_masks[0][:,::4,::4]
            with torch.no_grad():
                # gt_semantic_seg_Nx[0] [1,200,400], 
                # gt_masks_4x [#bbox,200,400]
                panoptic_gt = self.mask_matching(
                        gt_semantic_seg_Nx[0], gt_masks_4x)
                panoptic_gt = panoptic_gt.long()

            panoptic_loss = F.cross_entropy(
                    panoptic_logits, panoptic_gt, ignore_index = 255)
            pano_loss = {'loss_pano': panoptic_loss * self.train_cfg.loss_pano_weight}
            losses.update(pano_loss)

        return losses


    # ##################### REFERENCE FRAME ####################
    # def forward_ref_train(self,
    #                       x,
    #                       gt_bboxes=None,
    #                       gt_labels=None,
    #                       ):
    #     # Panoptic head forward ONLY
    #     fcn_output, fcn_score = self.panopticFPN(x[0:self.panopticFPN.num_levels])
    #     if gt_bboxes is not None and gt_labels is not None:
    #         gt_rois = bbox2roi(gt_bboxes) # [#bbox, 5]
    #         cls_idx = gt_labels[0] # [#bbox] / batch_size must be 1
    #         # fcn_score # [1,20,200,400]
    #         # compute mask logits with gt rois
    #         mask_feats = self.mask_roi_extractor(
    #                         x[:self.mask_roi_extractor.num_inputs], gt_rois)
    #                         # [#bbox,256,14,14]
    #         mask_score = self.mask_head(mask_feats) # [#bbox,#things+1,28,28], #things+1 =9
    #         nobj,_,H,W = mask_score.shape
    #         mask_score = mask_score.gather(1, cls_idx.view(-1,1,1,1).expand(-1,-1,H,W)) # [#bbox,1,28,28]
    #         # compute panoptic FCN logits: in a FCN format !
    #         mask_logits = self.mask_term(mask_score, gt_rois, cls_idx, fcn_score)
    #         mask_fcn_logits = self.mask_fcn_term(mask_score, gt_rois, cls_idx, fcn_score)
    #         panoptic_fcn_logits = fcn_score + mask_fcn_logits
    #         return fcn_score, mask_logits, panoptic_fcn_logits

    #     return fcn_output, fcn_score

    ##################### TEST HELPERS ####################
    # #########################################################
    def simple_test_bboxes_flow(self,
                               x,
                               img_meta,
                               proposals,
                               rcnn_test_cfg,
                               im_info,
                               rescale=False,
                               is_panoptic=True,
                               fcn_score=None,
                               flow=None):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        fid = int(img_meta[0]['filename'].split('_')[-1].split('.jpg')[0])
        # is_first = img_meta[0]['is_first']
        is_first = (fid==1)
        #### For "mask_roi_panoptic"
        cls_prob = F.softmax(cls_score, dim=1)
        if is_first or (not is_first and self.prev_bboxes is None):
            cls_prob, det_rois, cls_idx = self.mask_roi_panoptic(rois,
                bbox_pred, cls_prob, im_info)
        else:
            cls_prob, det_rois, cls_idx = self.mask_roi(rois, bbox_pred,
                cls_prob, im_info)

        det_labels = cls_idx - 1
        det_roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], det_rois)
        det_bboxes = roi2bbox(det_rois)[0]

        #### MASK LOGIT
        mask_feats = self.mask_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], det_rois)
        mask_score = self.mask_head(mask_feats) # [#,#things+1,28,28], #things+1=9
        nobj,_,H,W = mask_score.shape
        mask_score = mask_score.gather(1, cls_idx.view(-1,1,1,1).expand(-1,-1,H,W)) # [#,1,28,28]
        # sigmoid ==> [0,1]
        mask_score = mask_score.sigmoid()
        assert fcn_score is not None, "mask_logits needs fcn_score."
        mask_logits = self.mask_term(mask_score, det_rois, cls_idx, fcn_score)
        # [1,#,H/4,H/4]
        #### original branch
        if det_bboxes.nelement()==0:
            det_obj_ids=np.array([], dtype=np.int64)
            if is_first:
                self.prev_bboxes =  None
                self.prev_roi_feats = None
                self.prev_det_labels = None
                self.prev_mask_logits = None
                self.prev_nobj = None
            return det_bboxes, det_labels, det_obj_ids

        # recompute bbox match feature
        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.size(0))
            best_match_scores = np.ones((det_bboxes.size(0))) * (-100) # 
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = det_roi_feats
            self.prev_det_labels = det_labels
            self.prev_mask_logits = mask_logits
        else:
            assert self.prev_roi_feats is not None
            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)] # eg. 4
            prev_bbox_img_n = [self.prev_roi_feats.size(0)] # eg. 5
            match_score = self.track_head(det_roi_feats, self.prev_roi_feats, bbox_img_n, prev_bbox_img_n)[0] # [4]
            match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1,1)).float()
            bbox_ious = bbox_overlaps(det_bboxes[:,:4], self.prev_bboxes[:,:4])
            #### warp mask iou
            warp_mask_logits = self.flow_warping(self.prev_mask_logits, flow)
            # update after warping
            self.prev_mask_logits = warp_mask_logits
            mask_binary = mask_logits[0] >= 0.5 # [20,H,W]
            warp_binary = warp_mask_logits[0] >= 0.5 # [19,H,W]
            mask_binary = mask_binary.unsqueeze(1).expand((-1,self.prev_bboxes.size(0),-1,-1))
            warp_binary = warp_binary.unsqueeze(0).expand((nobj,-1,-1,-1))
            intersection = (mask_binary * warp_binary).sum((-2,-1))
            union = (mask_binary.sum((-2,-1)) + warp_binary.sum((-2,-1)) - intersection)
            warp_mask_ious = intersection.float() / union.float()

            # compute comprehensive score 
            comp_scores = self.track_head.compute_comp_scores(match_logprob, 
                # det_bboxes[:,4].view(-1, 1),
                cls_prob.view(-1,1),
                warp_mask_ious,
                label_delta,
                add_bbox_dummy=True) # [5]

            match_likelihood, match_ids = torch.max(comp_scores, dim =1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object, 
            # add tracking features/bboxes of new object
            match_likelihood = match_likelihood.cpu().numpy()
            match_ids = match_ids.cpu().numpy().astype(np.int32) # [4]
            det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1) # [-1, -1, -1, -1]
            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100) # 
            best_match_ids = np.ones((self.prev_bboxes.size(0)), dtype=np.int32) * (-1) # [-100, -100, -100, -100, -100]

            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    # add new object
                    # NOTE. obj_ids is one of [0,1,...5], 
                    # but is deducted by 1 later at "else". 
                    # So, assign "5" makes sense.
                    det_obj_ids[idx] = self.prev_roi_feats.size(0) # [5]
                    self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_mask_logits = torch.cat((self.prev_mask_logits, mask_logits[:,idx][:,None]), dim=1)
                    self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score 
                    obj_id = match_id - 1
                    # match_score = comp_scores[idx, match_id]
                    match_score = match_likelihood[idx]
                    if match_score > best_match_scores[obj_id]:
                        det_obj_ids[idx] = obj_id
                        # if matched before, undo
                        if best_match_ids[obj_id] >= 0: 
                            det_obj_ids[best_match_ids[obj_id]] = -1
                        best_match_scores[obj_id] = match_score
                        best_match_ids[obj_id] = idx
                        # udpate feature
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]
                        self.prev_mask_logits[:,obj_id] = mask_logits[:,idx]

            #### Now, all the proposals considered REDUNDANT are marked "-1"
            # Here, We assign "NEW" label to all these proposals.
            for idx, det_obj_id in enumerate(det_obj_ids):
                if det_obj_id >= 0:
                    continue
                det_obj_ids[idx] = self.prev_roi_feats.size(0) # [5]
                self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                self.prev_mask_logits = torch.cat((self.prev_mask_logits, mask_logits[:,idx][:,None]), dim=1)
                self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)

        return det_bboxes, det_labels, det_obj_ids, best_match_scores, cls_score, bbox_pred, cls_prob, det_rois, cls_idx 


    def simple_test_bboxes(self,
                           x,
                           img_meta,
                           proposals,
                           rcnn_test_cfg,
                           im_info,
                           rescale=False,
                           is_panoptic=True):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        cls_score, bbox_pred = self.bbox_head(roi_feats)
        img_shape = img_meta[0]['img_shape']
        scale_factor = img_meta[0]['scale_factor']
        ### for VIPER:
        if 'city' in img_meta[0]['filename']:
            #### for CITYSCAPES_EXT:
            if 'iid' in img_meta[0]:
                iid = img_meta[0]['iid']
                # vid = iid // 10000
                fid = iid % 10000
                is_first = (fid == 1)
            else:
                vid = int(img_meta[0]['filename'].split('/')[-1].split('_')[0])
                fid = int(img_meta[0]['filename'].split('/')[-1].split('_')[1])
                is_first = (fid == vid*6 - 5)
        else:
            fid = int(img_meta[0]['filename'].split('_')[-1].split('.jpg')[0])
            is_first = (fid==1)
        
        #### For "mask_roi_panoptic"
        cls_prob = F.softmax(cls_score, dim=1)
        cls_prob, det_rois, cls_idx = self.mask_roi_panoptic(rois,
            bbox_pred, cls_prob, im_info)

        det_labels = cls_idx - 1
        det_roi_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], det_rois)
        det_bboxes = roi2bbox(det_rois)[0]

        if det_bboxes.nelement()==0:
            det_obj_ids=np.array([], dtype=np.int64)
            if is_first:
                self.prev_bboxes =  None
                self.prev_roi_feats = None
                self.prev_det_labels = None
            return det_bboxes, det_labels, det_obj_ids

        # recompute bbox match feature
        if is_first or (not is_first and self.prev_bboxes is None):
            det_obj_ids = np.arange(det_bboxes.size(0))
            best_match_scores = np.ones((det_bboxes.size(0))) * (-100) # 
            # save bbox and features for later matching
            self.prev_bboxes = det_bboxes
            self.prev_roi_feats = det_roi_feats
            self.prev_det_labels = det_labels
        else:
            assert self.prev_roi_feats is not None
            # only support one image at a time
            bbox_img_n = [det_bboxes.size(0)] # eg. 4
            prev_bbox_img_n = [self.prev_roi_feats.size(0)] # eg. 5
            match_score = self.track_head(det_roi_feats, self.prev_roi_feats, bbox_img_n, prev_bbox_img_n)[0] # [4]
            match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)
            label_delta = (self.prev_det_labels == det_labels.view(-1,1)).float()
            bbox_ious = bbox_overlaps(det_bboxes[:,:4], self.prev_bboxes[:,:4])
            # compute comprehensive score 
            comp_scores = self.track_head.compute_comp_scores(match_logprob, 
                # det_bboxes[:,4].view(-1, 1),
                cls_prob.view(-1,1),
                bbox_ious,
                label_delta,
                add_bbox_dummy=True) # [5]

            match_likelihood, match_ids = torch.max(comp_scores, dim =1)
            # translate match_ids to det_obj_ids, assign new id to new objects
            # update tracking features/bboxes of exisiting object, 
            # add tracking features/bboxes of new object
            match_likelihood = match_likelihood.cpu().numpy()
            match_ids = match_ids.cpu().numpy().astype(np.int32) # [4]
            det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1) # [-1, -1, -1, -1]
            best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100) # 
            best_match_ids = np.ones((self.prev_bboxes.size(0)), dtype=np.int32) * (-1) # [-100, -100, -100, -100, -100]

            for idx, match_id in enumerate(match_ids):
                if match_id == 0:
                    # add new object
                    # NOTE. obj_ids is one of [0,1,...5], 
                    # but is deducted by 1 later at "else". 
                    # So, assign "5" makes sense.
                    det_obj_ids[idx] = self.prev_roi_feats.size(0) # [5]
                    self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                    self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                    self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                else:
                    # multiple candidate might match with previous object, here we choose the one with
                    # largest comprehensive score 
                    obj_id = match_id - 1
                    # match_score = comp_scores[idx, match_id]
                    match_score = match_likelihood[idx]
                    if match_score > best_match_scores[obj_id]:
                        det_obj_ids[idx] = obj_id
                        # if matched before, undo
                        if best_match_ids[obj_id] >= 0: 
                            det_obj_ids[best_match_ids[obj_id]] = -1
                        best_match_scores[obj_id] = match_score
                        best_match_ids[obj_id] = idx
                        # udpate feature
                        self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                        self.prev_bboxes[obj_id] = det_bboxes[idx]

            #### Now, all the proposals considered REDUNDANT are marked "-1"
            # Here, We assign "NEW" label to all these proposals.
            for idx, det_obj_id in enumerate(det_obj_ids):
                if det_obj_id >= 0:
                    continue
                det_obj_ids[idx] = self.prev_roi_feats.size(0) # [5]
                self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[idx][None]), dim=0)
                self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
            
        return det_bboxes, det_labels, det_obj_ids, best_match_scores, cls_score, bbox_pred, cls_prob, det_rois, cls_idx 


    def simple_test_mask(self,
                         x,
                         img_meta,
                         det_bboxes,
                         det_labels,
                         det_obj_ids=None,
                         rescale=False):
        # image shape of the first image in the batch (only one)
        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']
        if det_bboxes.shape[0] == 0 or True:
            segm_result = [[] for _ in range(self.mask_head.num_classes - 1)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            _bboxes = (det_bboxes[:, :4] * scale_factor
                       if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_feats = self.mask_roi_extractor(
                x[:len(self.mask_roi_extractor.featmap_strides)], mask_rois)
            mask_pred = self.mask_head(mask_feats)
            segm_result = self.mask_head.get_seg_masks(
                mask_pred, _bboxes, det_labels, self.test_cfg.rcnn, ori_shape,
                scale_factor, rescale, det_obj_ids=det_obj_ids)
        return segm_result


    ########################### TEST #######################
    def simple_test(self, img, img_meta, proposals=None, rescale=False,
                        ref_img=None, gt_flow=None):
        im_info = np.array([[float(img.shape[2]), float(img.shape[3]), 1.0]])
        # This has not been handled in base.py ...
        if ref_img is not None:
            ref_img=ref_img[0]
        if gt_flow is not None:
            gt_flow=gt_flow[0]
        ####
        img_clone = img.clone().detach()
        ref_img_clone = ref_img.clone().detach()

        if hasattr(self.test_cfg, 'flownet2'):
            flowR2T, occ_mask = self.compute_flow(img.clone(), ref_img.clone(), scale_factor=0.25)
        else:
            flowR2T = None

        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."
        assert self.with_track, "Track head must be implemented"
        assert proposals is None
        
        x = self.extract_feat(img)
        ref_x = self.extract_feat(ref_img)

        # x, flow_fine = self.extra_neck(x, ref_x, flowR2T)
        x = self.extra_neck(x, ref_x, flowR2T)

        assert hasattr(self, 'panopticFPN') and self.panopticFPN is not None, \
                "FCN HEAD must be implemented."
        fcn_output, fcn_score = self.panopticFPN(x[0:self.panopticFPN.num_levels])

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) 

        det_bboxes, det_labels, det_obj_ids, best_match_scores, cls_score, bbox_pred, cls_prob, mask_rois, cls_idx =  self.simple_test_bboxes(
                x, img_meta, proposal_list, self.test_cfg.rcnn, im_info,
                rescale=rescale, 
                is_panoptic=True)   
        # det_bboxes, det_labels, det_obj_ids, best_match_scores, cls_score, bbox_pred, cls_prob, mask_rois, cls_idx = self.simple_test_bboxes_flow(
        #         x, img_meta, proposal_list, self.test_cfg.rcnn, im_info,
        #         rescale=rescale, 
        #         is_panoptic=True,
        #         fcn_score=fcn_score,
        #         flow=flowR2T)

        bbox_results = bbox2result_with_id(det_bboxes, det_labels,
            det_obj_ids, self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, 
                rescale=rescale, det_obj_ids=det_obj_ids)

            #### test with panoptic segm
            if hasattr(self, 'panopticFPN') and self.panopticFPN is not None:
                if hasattr(self.test_cfg, 'loss_pano_weight'):
                    # fcn_output, fcn_score = self.panopticFPN(x[0:self.panopticFPN.num_levels])

                    mask_feats = self.mask_roi_extractor(
                            x[:self.mask_roi_extractor.num_inputs], mask_rois)
                            # [#bbox,256,14,14]
                    mask_score = self.mask_head(mask_feats) # [#bbox,#things+1,28,28], #things+1=9
                    nobj,_,H,W = mask_score.shape
                    mask_score = mask_score.gather(1, cls_idx.view(-1,1,1,1).expand(-1,-1,H,W)) # [#bbox,1,28,28]
                    # compute panoptic logits
                    keep_inds, mask_logits = self.mask_removal(mask_rois[:,1:], cls_prob,mask_score, cls_idx, fcn_output.shape[2:])
                    # convert det_obj_ids to torch tensor
                    det_obj_ids = torch.from_numpy(det_obj_ids).to(cls_idx.device)
                    # mask logits [1,k,1024,2048]
                    mask_rois = mask_rois[keep_inds] #[k,5]
                    cls_idx = cls_idx[keep_inds] #[k]
                    det_labels = det_labels[keep_inds] #[k]
                    det_obj_ids = det_obj_ids[keep_inds] #[k]
                    cls_prob = cls_prob[keep_inds] #[k]
                    #### filter out redundant detection
                    # best_inds = torch.index_select(det_obj_ids>=0, 0, torch.arange(0,det_obj_ids.size(0)).cuda())
                    # mask_rois = mask_rois[best_inds] #[k',5]
                    # cls_idx = cls_idx[best_inds] #[k']
                    # det_labels = det_labels[best_inds] #[k']
                    # det_obj_ids = det_obj_ids[best_inds] #[k']
                    # cls_prob = cls_prob[best_inds] #[k']
                    # mask_logits = mask_logits[:,best_inds] # [k']

                    # get semantic segm logits
                    seg_stuff_logits, seg_inst_logits = self.seg_term(cls_idx, fcn_output, mask_rois*4.0)
                    # seg_stuff_logits [1,11,1024,2048]
                    # seg_inst_logits [1,k,1024,2048]
                    panoptic_logits = torch.cat([seg_stuff_logits, 
                            (seg_inst_logits + mask_logits)], dim=1)
                    panoptic_output = torch.max(F.softmax(panoptic_logits, dim=1), dim=1)[1]
                    # fcn_output [1,1024,2048]
                    fcn_output = torch.max(F.softmax(fcn_output, dim=1), dim=1)[1]

                    #### crop back into original input shape
                    img_shape_withoutpad = img_meta[0]['img_shape']
                    fcn_output = fcn_output[:,0:img_shape_withoutpad[0],0:img_shape_withoutpad[1]]
                    panoptic_output = panoptic_output[:,0:img_shape_withoutpad[0],0:img_shape_withoutpad[1]]

                    pano_results = {
                        'fcn_outputs': fcn_output,
                        'panoptic_cls_inds': cls_idx,
                        'panoptic_cls_prob': cls_prob,
                        'panoptic_det_labels': det_labels,
                        'panoptic_det_obj_ids': det_obj_ids,
                        'panoptic_outputs': panoptic_output,
                    }
                    return bbox_results, segm_results, pano_results
                # normal
                else:
                    semantic_results = self.simple_test_semantic_segm(x, img_meta)
                    return bbox_results, segm_results, semantic_results
                
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False,
                    img_ref=None, gt_flow=None):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs, img_ref=img_ref, gt_flow=gt_flow), 
                    img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs, img_ref=img_ref, gt_flow=gt_flow),
                    img_metas, proposal_list, self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs, img_ref=img_ref, gt_flow=gt_flow), 
                    img_metas, det_bboxes, det_labels)
            # test with panoptic segm
            if hasattr(self, 'panopticFPN') and self.panopticFPN is not None:
                semantic_results = self.aug_test_semantic_segm(self.extract_feats(imgs), img_metas)
                return bbox_results, segm_results, semantic_results

            return bbox_results, segm_results
        else:
            return bbox_results
