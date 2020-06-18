--------------------------------------------
Modified files for VPSNet - Video Panoptic Segmentation
Based on mmdetection

mmdet
    models
        detectors
            panoptic_fusetrack.py
            panoptic_fuse.py
            test_mixins.py
            two_stage.py
            base.py
        extra_necks
            bfp_tcea.py
        flow_modules/*
        losses
            smooth_l1_loss.py
            cross_entropy_loss.py

        panoptic/*
        track_heads
            track_head.py
        utils/*
        builder.py

    datasets
        pipelines/*
        cityscapes.py
        cityscapes_vps.py
        coco.py
        custom.py

    core
        bbox/*

    apis
        train.py

    ops/












