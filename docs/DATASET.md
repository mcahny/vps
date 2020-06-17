## Dataset

### Disclaimer
This software is for non-commercial use only. The source code is released under the Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) Licence. Permission is granted to use the data given that you agree to both [our license terms](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) and that of the original [Cityscapes-dataset](https://cityscpaes-dataset.com/license/)

### Download datasets
a. Symlink the `$DATA_ROOT` dataset to `$MMDETECTION/data` folder. 
b. Download `leftImg8bit_sequence.zip` and `gtFine.zip` from the [Cityscapes-dataset webpage](https://cityscpaes-dataset.com/) in the `data` folder. You need only `val/` of these datasets to construct Cityscapes-VPS.
c. Download Cityscapes-VPS from here in `$CITY_VPS` folder.

### Merge Cityscapes and Cityscapes-VPS datasets
Fetch cityscapes sequence images into `$CITY_VPS/SPLIT/img_all` and merge two datasets at `$CITY_VPS/SPLIT/cls` and `$CITY_VPS/SPLIT/inst`.
```
# SPLIT = 'val' or 'test'
python prepare_city_vps/fetch_city_images.py --src_dir data/leftImg8bit_sequence/val/ \
    --dst_dir $CITY_VPS --mode SPLIT 
python prepare_city_vps/merge_datasets.py --src_dir data/gtFine/val/ \
    --dst_dir $CITY_VPS --mode SPLIT
```

### Create panoptic labels
Create `labelmap/`, `panoptic_inst/`, and `panoptic_video/` in `$CITY_VPS/SPLIT/` by running following commands.
```
python prepare_city_vps/create_panoptic_labels.py --root_dir $CITY_VPS --mode SPLIT
python prepare_city_vps/create_panoptic_video_labels.py --root_dir $CITY_VPS --mode SPLIT
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
│   │   │   ├── img
│   │   │   ├── img_all
```

-----------------------------


```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── cityscapes
│   │   ├── annotations
│   │   │   ├── instancesonly_gtFine_train.json
│   │   │   ├── instancesonly_pano_gtFine_val.json
│   │   │   ├── cityscapes_fine_val.json
│   │   ├── panoptic
│   │   ├── train
│   │   ├── train_nbr
│   │   ├── labels
│   │   ├── val
│   │   ├── val_nbr
│   ├── cityscapes_vps
│   │   ├── instances_train_01_city_coco_rle.json
│   │   ├── instances_val_01_city_coco_rle.json
│   │   ├── instances_val_01_im_info.json
│   │   ├── cityscapes_ext_panoptic_val_video.json
│   │   ├── train
│   │   │   ├── img
│   │   │   ├── labelmap
│   │   ├── val
│   │   │   ├── img_all
│   │   │   ├── panoptic_video_vivid
```




Note:

1. The git commit id will be written to the version number with step d, e.g. 0.6.0+2e7045c. The version will also be saved in trained models.
It is recommended that you run step d each time you pull some updates from github. If C/CUDA codes are modified, then this step is compulsory.

2. Following the above instructions, mmdetection is installed on `dev` mode, any local modifications made to the code will take effect without the need to reinstall it (unless you submit some commits and want to update the version number).

### Another option: Docker Image

We provide a [Dockerfile](../docker/Dockerfile) to build an image.

```shell
# build an image with PyTorch 1.1, CUDA 10.0 and CUDNN 7.5
docker build -t mmdetection docker/
```

### Prepare datasets

It is recommended to symlink the dataset root to `$MMDETECTION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── cityscapes
│   │   ├── annotations
│   │   ├── train
│   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012

```
The cityscapes annotations have to be converted into the coco format using the [cityscapesScripts](https://github.com/mcordts/cityscapesScripts) toolbox.
We plan to provide an easy to use conversion script. For the moment we recommend following the instructions provided in the 
[maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark/tree/master/maskrcnn_benchmark/data) toolbox. When using this script all images have to be moved into the same folder. On linux systems this can e.g. be done for the train images with:
```shell
cd data/cityscapes/
mv train/*/* train/
```

### Scripts

[Here](https://gist.github.com/hellock/bf23cd7348c727d69d48682cb6909047) is
a script for setting up mmdetection with conda.

### Multiple versions

If there are more than one mmdetection on your machine, and you want to use them alternatively, the recommended way is to create multiple conda environments and use different environments for different versions.

Another way is to insert the following code to the main scripts (`train.py`, `test.py` or any other scripts you run)
```python
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
```
or run the following command in the terminal of corresponding folder.
```shell
export PYTHONPATH=`pwd`:$PYTHONPATH
```
