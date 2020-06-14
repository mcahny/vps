
# VPSNet for Video Panoptic Segmentation

## Introduction
Official pytorch implementation for "Video Panoptic Segmentation" (CVPR 2020 Oral)

[Dahun Kim](https://mcahny.github.io/), [Sanghyun Woo](https://sites.google.com/view/sanghyunwoo/), [Joon-Young Lee](https://joonyoung-cv.github.io/), and [In So Kweon](https://rcv.kaist.ac.kr).
[[Paper](https://arxiv.org/abs/1905.01639)] [[Project page](https://sites.google.com/view/deepvinet/)] [[Video results](https://youtu.be/RtThGNTvkjY)]  

```bibtex
@inproceedings{kim2020vps,
  title={Video Panoptic Segmentation},
  author={Dahun Kim and Sanghyun Woo and Joon-Young Lee and In So Kweon},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

<img src="./image/panoptic_pair_240.gif" width="800"> 

Image-level baseline (left) / **VPSNet** result (right)

## Disclaimer
This repo is tested under Python 3.7, PyTorch 1.4 and mmcv==0.2.14.

## Installation
a. This repo is built based on [mmdetection](https://github.com/open-mmlab/mmdetection) commit hash `4357697`. Please refer to [INSTALL.md](INSTALL.md) to install the library.
You can use following commands to create conda env with related dependencies.
```
conda create -n vps python=3.7 -y
conda activate vps
conda install -c pytorch pytorch=1.4 torchvision -y
pip install -r requirements.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install "git+https://github.com/cocodataset/panopticapi.git"
pip install -v -e . 

```
b. You also need to install dependencies for [Flownet2](https://github.com/NVIDIA/flownet2-pytorch) and [UPSNet](https://github.com/uber-research/UPSNet) modules. 
```
bash ./init_flownet.sh
bash ./init_upsnet.sh
```

c. You may also need to download some pretrained weights.
```
bash ./download_weights.sh
```

## Dataset
1. Download Cityscapes-VPS from here.
2. Symlink the `$DATA_ROOT` dataset to `$MMDETECTION/data` folder. 
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
│   ├── cityscapes_ext
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

## Testing
Our trained model is available for download at Google Drive. Run the following command to test the model on Cityscapes and Cityscapes-VPS.
```
# A. Test Image Panoptic Quality on Cityacapes.
python tools/test_eval_ipq.py configs/cityscapes/ups_pano_flow_tcea.py ./work_dirs/cityscapes/ups_pano_flow_tcea_vp/latest.pth --out work_dirs/cityscapes/ups_pano_flow_tcea_vp/val.pkl --dataset Cityscapes --name val --gpus 0

# B. Test Video Panoptic Quality (VPQ) on Cityscapes-VPS.
python tools/test_vpq.py configs/cityscapes/ups_pano_ext_track_flow_tcea.py work_dirs/cityscapes_ext/ups_pano_ext_fusetrack_vpct/latest.pth --out work_dirs/cityscapes_ext/ups_pano_ext_fusetrack_vpct/val_unified_.pkl --name val --dataset CityscapesExt --txt_dir val --has_track --n_video 100
python tools/eval_vpq.py configs/cityscapes/ups_pano_ext_track_flow_tcea.py work_dirs/cityscapes_ext/ups_pano_ext_fusetrack_vpct/latest.pth --out work_dirs/cityscapes_ext/ups_pano_ext_fusetrack_vpct/val_unified_.pkl --name val --dataset CityscapesExt --txt_dir val --has_track --n_video 100
```
A json file containing the predicted result will be generated as `pred.json` and `pan/*.png` at  `work_dirs/cityscapes_ext/ups_pano_ext_fusetrack_vpct/val_pans_unified/`. Cityscapes-VPS currently only allows evaluation on the codalab server. Please upload the generated result to codalab server to see actual performances.


## Training
You can use following commands to train Fuse and FuseTrack models.
```
# A. Train Fuse model on image-level Cityscapes.
bash ./tools/dist_train.sh configs/cityscapes/ups_pano_flow_tcea.py 8

# B. Train FuseTrack model on video-level Cityscapes-VPS.
bash ./tools/dist_train.sh configs/cityscapes/ups_pano_ext_track_flow_tcea.py 8
```

## Evaluation
Cityscapes-VPS `test` split only allows evaluation on the codalab server. Please upload `submission.zip` to codalab server to see actual performances.

```
submission.zip
├── pred.json
├── pan_pred.zip
│   ├── 0005_0025_frankfurt_000000_001736.png
│   ├── 0005_0026_frankfurt_000000_001741.png
│   ├── ...
│   ├── 0500_3000_munster_000173_000029.png
```

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibtex
@inproceedings{kim2020vps,
  title={Video Panoptic Segmentation},
  author={Dahun Kim and Sanghyun Woo and Joon-Young Lee and In So Kweon},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
## Terms of Use

This software is for non-commercial use only.
The source code is released under the Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) Licence
(see [this](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for details)

## Contact

If you have any questions regarding the repo, please contact Dahun Kim (mcahny01@gmail.com) or create an issue.
