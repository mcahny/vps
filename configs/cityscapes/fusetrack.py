# model settings
model = dict(
    type='PanopticTrackFlowTcea',
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    extra_neck=dict(
        type='BFPTcea',
        in_channels=256,
        num_levels=5,
        refine_level=0,
        refine_type='conv',
        center=0,
        nframes=2),
    panoptic = dict(
            type='UPSNetFPN',
            in_channels=256,
            out_channels=128,
            num_levels=4,
            num_things_classes=8,
            num_classes=19,
            ignore_label=255,
            loss_weight=1.0), 
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=9,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
    track_head=dict(
        type='TrackHead',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        match_coeff=[1.0,2.0, 10],
        loss_match=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
        ),
    mask_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    mask_head=dict(
        type='FCNMaskHead',
        num_convs=4,
        in_channels=256,
        conv_out_channels=256,
        num_classes=9,
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        mask_size=28,
        pos_weight=-1,
        debug=False),
    loss_pano_weight=0.5,
    flownet2=[],
    class_mapping = {1:11, 2:12, 3:13, 4:14, 5:15, 6:16, 7:17, 8:18})
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5),
    loss_pano_weight=None,
    flownet2=[],
    class_mapping = {1:11, 2:12, 3:13, 4:14, 5:15, 6:16, 7:17, 8:18})
# dataset settings
dataset_type = 'CityscapesVideoOfsDataset'
data_root = 'data/cityscapes_vps/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadRefImageFromFile', span=[0]),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, 
        with_seg=True, with_pid=True,
        semantic2label={0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9,
                        10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16,
                        17:17, 18:18, -1:255, 255:255},),
    dict(type='Resize', img_scale=[(2048, 1024)], keep_ratio=True,
        multiscale_mode='value', ratio_range=(0.8, 1.5)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomCrop', crop_size=(800, 1600)),
    dict(type='Pad', size_divisor=32),
    dict(type='SegResizeFlipCropPadRescale', scale_factor=[1, 0.25]),
    dict(type='DefaultFormatBundle'),

    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 
            'gt_obj_ids', 'gt_masks', 'gt_semantic_seg', 
            'gt_semantic_seg_Nx', 'ref_img', 'ref_bboxes', 
            'ref_labels', 'ref_obj_ids', 'ref_masks']),
]
test_pipeline = [
    dict(type='LoadRefImageFromFile'),

    dict(
        type='MultiScaleFlipAug',
        img_scale=[(2048, 1024)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'ref_img']),
            dict(type='Collect', keys=['img', 'ref_img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        # times=1,
        times=8,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
            'instances_train_city_vps_rle.json',
            img_prefix=data_root + 'train/img/',
            ref_prefix=data_root + 'train/img/',
            seg_prefix=data_root + 'train/labelmap/',
            pipeline=train_pipeline,
            ref_ann_file=data_root + 
            'instances_train_city_vps_rle.json',
            offsets=[-1,+1])),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'instances_val_city_vps_rle.json',
        img_prefix=data_root + 'val/img/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        # 'im_all_info_val_city_vps.json',
        # img_prefix=data_root + 'val/img_all/',
        # ref_prefix=data_root + 'val/img_all/',
        'im_all_info_test_city_vps.json',
        img_prefix=data_root + 'test/img_all/',
        ref_prefix=data_root + 'test/img_all/',
        nframes_span_test=30,
        pipeline=test_pipeline))
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8,11])
checkpoint_config = dict(interval=4)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'

work_dir = './work_dirs/cityscapes_vps/fusetrack_vpct'
load_from = './work_dirs/cityscapes/fuse_vpct/latest.pth'
resume_from = None
workflow = [('train', 1)]




