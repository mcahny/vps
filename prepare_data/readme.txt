
[For release]
    im_all_info_val_city_vps.json (all dense frames info for inference)
    im_all_info_test_city_vps.json (all dense frames info for inference)
    
    instances_train_city_vps_rle.json
    instances_val_city_vps_rle.json (actually, no validation during training)

    panoptic_im_train_city_vps.json
    panoptic_im_val_city_vps.json
    panoptic_im_test_city_vps.json

    train_gt.zip
    val_gt.zip


[Created by the code]
    train/labelmap/*.png (for VPSNet training)
    train/panoptic_inst/*.png (byproduct, not used)
    train/panoptic_video/*.png (byproduct, not used)

    val/labelmap/*.png (byproduct, not used)
    val/panoptic_inst/*.png (byproduct, not used)
    val/panoptic_video/*.png (for VPQ eval)
    
    panoptic_gt_val_city_vps.json (for VPQ eval)


-------------------- Descriptions ------------------

[Image files]
    * Ours
        train_gt.zip
        val_gt.zip


[Json files]
    * image info json files for train/val/test split (All needed)
        panoptic_im_train_city_vps.json
        panoptic_im_val_city_vps.json
        panoptic_im_test_city_vps.json 

    * For VPSNet training
        instances_train_city_vps_rle.json
        instances_val_city_vps_rle.json (actually, no validation during training)

    * For VPSNet inference
        im_all_info_val_city_vps.json (all dense frames info for inference)
        im_all_info_test_city_vps.json (all dense frames info for inference)

    * For VPQ evaluation
        panoptic_gt_val_city_vps.json (created by code)
        

[Not used]
    test_gt.zip (not be released)
    instances_test_city_vps_rle.json (not be released)
    panoptic_gt_test_city_vps.json (created by code, not be released)



