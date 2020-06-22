
# VPSNet for Video Panoptic Segmentation

## Official pytorch implementation for "Video Panoptic Segmentation" <br/> (CVPR 2020 Oral)  [[Paper](https://drive.google.com/file/d/1jJkwTMFRNEx-ka9u1GZastLxctqOwN8t)] [[Dataset](https://www.dropbox.com/s/ecem4kq0fdkver4/cityscapes-vps-dataset-1.0.zip?dl=0)] [[Project](https://sites.google.com/view/video-panoptic)] [[Slides](https://drive.google.com/uc?id=1525wpf3kDy2kEsbaPo79ak713gYSfBPe)]

[Dahun Kim](https://mcahny.github.io/), [Sanghyun Woo](https://sites.google.com/view/sanghyunwoo/), [Joon-Young Lee](https://joonyoung-cv.github.io/), and [In So Kweon](https://rcv.kaist.ac.kr).

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
This repo is tested under Python 3.7, PyTorch 1.4, Cuda 10.0, and mmcv==0.2.14.

## Installation
a. This repo is built based on [mmdetection](https://github.com/open-mmlab/mmdetection) commit hash `4357697`. Our modifications for VPSNet implementation are listed [here](mmdet/readme.txt). Please refer to [INSTALL.md](docs/INSTALL.md) to install the library.
You can use following commands to create conda env with related dependencies.
```
conda create -n vps python=3.7 -y
conda activate vps
conda install pytorch=1.4 torchvision cudatoolkit=10.0 -c pytorch -y
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
pip install gdown
bash ./download_weights.sh
```

## Dataset
You can [download Cityscapes-VPS here](https://www.dropbox.com/s/ecem4kq0fdkver4/cityscapes-vps-dataset-1.0.zip?dl=0). It provides 2500-frame panoptic labels that temporally extend the 500 Cityscapes image-panoptic labels. There are total 3000-frame panoptic labels which correspond to 5, 10, 15, 20, 25, and 30th frames of each 500 videos, where all instance ids are associated over time.
<img src="./image/dataset.png" width="1000"> 

It not only supports video panoptic segmentation (VPS) task, but also provides super-set annotations for video semantic segmentation (VSS) and video instance segmentation (VIS) tasks. 

Necessary data for Cityscapes-VPS training, testing, and evaluation are as follows.
Please refer to [DATASET.md](docs/DATASET.md) for dataset preparation. 
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

## Evaluation Metric 
<img src="./image/vpq_metric.png" width="800"> 


## Testing
Our trained models are available for download [here](https://drive.google.com/uc?id=1KcHYnghbs2KC6hQc7QVkPkEiJMrLr73s). Rename it to `latest.pth` and run the following commands to test the model on Cityscapes-VPS.

* FuseTrack model for Video Panoptic Quality (VPQ) on Cityscapes-VPS `val` set (`vpq-λ.txt` will be saved.)
```
python tools/test_vpq.py configs/cityscapes/fusetrack.py \
  work_dirs/cityscapes_vps/fusetrack_vpct/latest.pth \
  --out work_dirs/cityscapes_vps/fusetrack_vpct/val.pkl \
  --pan_im_json_file data/cityscapes_vps/panoptic_im_val_city_vps.json \
  --n_video 50 \
python tools/eval_vpq.py \
  --submit_dir work_dirs/cityscapes_vps/fusetrack_vpct/val_pans_unified/ \
  --truth_dir data/cityscapes_vps/val/panoptic_video/ \
  --pan_gt_json_file data/cityscapes_vps/panoptic_gt_val_city_vps.json
```
* FuseTrack model VPS inference on Cityscapes-VPS `test` set
```
python tools/test_vpq.py configs/cityscapes/fusetrack.py \
  work_dirs/cityscapes_vps/fusetrack_vpct/latest.pth \
  --out work_dirs/cityscapes_vps/fusetrack_vpct/test.pkl \
  --dataset CityscapesVps --has_track --n_video 50 \
  --pan_im_json_file data/cityscapes_vps/panoptic_im_test_city_vps.json
```
Files containing the predicted results will be generated as `pred.json` and `pan_pred/*.png` at  `work_dirs/cityscapes_vps/fusetrack_vpct/test_pans_unified/`. 

Cityscapes-VPS `test` split currently only allows evaluation on the codalab server. Please upload `submission.zip` to codalab server (will be open soon) to see actual performances.
```
submission.zip
├── pred.json
├── pan_pred.zip
│   ├── 0005_0025_frankfurt_000000_001736.png
│   ├── 0005_0026_frankfurt_000000_001741.png
│   ├── ...
│   ├── 0500_3000_munster_000173_000029.png
```


## Training
* Train FuseTrack model on video-level Cityscapes-VPS. We start from [initial weights](https://drive.google.com/uc?id=1t69I1u0QKl-N4eciYv3UYXFOQSYb25cD) of image panoptic segmentation (IPS) model, pretrained on the original Cityscapes. Place it at `work_dirs/cityscapes/fuse_vpct/` and rename to `latest.pth` and run the following command.
```
# Multi-GPU distributed training
bash ./tools/dist_train.sh configs/cityscapes/fusetrack.py ${GPU_NUM}
# OR
python ./tools/train.py configs/cityscapes/fusetrack.py --gpus ${GPU_NUM}
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


## Acknowledgements
This project has used utility functions from other wonderful open-sourced libraries. We would especially thank the authors of:
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [UPSNet](https://gitub.com/uber-research/UPSNet)
* [FlowNet2](https://NVIDIA/flownet2-pytorch)



## Contact

If you have any questions regarding the repo, please contact Dahun Kim (mcahny01@gmail.com) or create an issue.
