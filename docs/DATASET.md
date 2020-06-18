## Dataset

### Disclaimer
This software is for non-commercial use only. The source code is released under the Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) Licence. Permission is granted to use the data given that you agree to both [our license terms](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) and that of the original [Cityscapes-dataset](https://cityscpaes-dataset.com/license/)

### Download datasets
a. Symlink the `$DATA_ROOT` dataset to `$MMDETECTION/data` folder. 

b. Download [Cityscapes-VPS here](https://www.dropbox.com/s/ecem4kq0fdkver4/cityscapes-vps-dataset-1.0.zip?dl=0) in `$CITY_VPS = data/cityscapes_vps/` folder.

c. Download `leftImg8bit_sequence.zip` and `gtFine.zip` from the [Cityscapes-dataset webpage](https://cityscpaes-dataset.com/) in `data` folder. You need only `val/` of these datasets to construct Cityscapes-VPS.


### Merge Cityscapes into Cityscapes-VPS
We have 2400/600/600 frames for train/val/test splits. Fetch cityscapes sequence images into `$CITY_VPS/SPLIT/img` and `$CITY_VPS/SPLIT/img_all`, and merge two datasets labels at `$CITY_VPS/SPLIT/cls` and `$CITY_VPS/SPLIT/inst`.
```
bash ./prepare_data/merge_datasets.sh \
    data/cityscapes_vps  data/leftImg8bit_sequence/val/  data/gtFine/val/
# OR
# For SPLIT = 'train'/'val'/'test'
python prepare_data/fetch_city_images.py \
    --src_dir data/leftImg8bit_sequence/val/ \
    --dst_dir $CITY_VPS --mode SPLIT 
python prepare_data/merge_datasets.py \
    --src_dir data/gtFine/val/ \
    --dst_dir $CITY_VPS --mode SPLIT
```

### Create video-panoptic labels
Create `labelmap/`, `panoptic_inst/`, and `panoptic_video/` in `$CITY_VPS/SPLIT/` by running following commands.
```
bash ./prepare_data/create_panoptic_labels.sh \
    data/cityscapes_vps/
# OR
# For SPLIT = 'train'/'val'
python prepare_data/create_panoptic_labels.py --root_dir $CITY_VPS --mode SPLIT
# For SPLIT = 'val'
python prepare_data/create_panoptic_video_labels.py --root_dir $CITY_VPS --mode SPLIT
```

### Directory Structure
Necessary data for training, testing, and evaluation are as follows. You may delete other data for disk usage.
```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── cityscapes_vps
│   │   ├── panoptic_im_train_city_vps.json
│   │   ├── panoptic_im_val_city_vps.json
│   │   ├── panoptic_im_test_city_vps.json  
│   │   ├── instances_train_city_vps_rle.json (for training)
│   │   ├── instances_val_city_vps_rle.json 
│   │   ├── im_all_info_val_city_vps.json (for inference)
│   │   ├── im_all_info_test_city_vps.json (for inference)
│   │   ├── panoptic_gt_val_city_vps.json (for VPQ eval)
│   │   ├── train 
│   │   │   ├── img
│   │   │   ├── labelmap
│   │   ├── val
│   │   │   ├── img
│   │   │   ├── img_all
│   │   │   ├── panoptic_video
│   │   ├── test
│   │   │   ├── img_all
```
