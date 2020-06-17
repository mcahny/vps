
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
a. This repo is built based on [mmdetection](https://github.com/open-mmlab/mmdetection) commit hash `4357697`. Please refer to [INSTALL.md](docs/INSTALL.md) to install the library.
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
Cityscapes-VPS dataset is also a super-set of video semantic segmentation and video instance segmentation.
Please refer to [DATASET.md](docs/DATASET.md) for dataset preparation.


## Testing
Our trained models are available for download at Google Drive. Run the following command to test the model on Cityscapes and Cityscapes-VPS.

a. Image Panoptic Quality on Cityscapes.
```
python tools/test_eval_ipq.py \
    configs/cityscapes/ups_pano_flow_tcea.py \
    work_dirs/cityscapes/ups_pano_flow_tcea_vp/latest.pth \
    --out work_dirs/cityscapes/ups_pano_flow_tcea_vp/val.pkl \
    --dataset Cityscapes --name val --gpus 0
```
b. Video Panoptic Quality (VPQ) on Cityscapes-VPS.
```
python tools/test_vpq.py \
  configs/cityscapes/ups_pano_ext_track_flow_tcea.py \
  work_dirs/cityscapes_ext/ups_pano_vps_fusetrack_vpct/latest.pth \
  --out work_dirs/cityscapes_ext/ups_pano_vps_fusetrack_vpct/val0615.pkl \
  --name val0615 --dataset CityscapesExt --has_track --n_video 50 \
  --pan_im_json_file data/cityscapes_ext/panoptic_im_val_city_vps.json --load

python tools/test_vpq.py configs/cityscapes/ups_pano_ext_track_flow_tcea.py work_dirs/cityscapes_ext/ups_pano_vps_fusetrack_vpct/latest.pth --out work_dirs/cityscapes_ext/ups_pano_vps_fusetrack_vpct/test0616.pkl --name test0616 --dataset CityscapesExt --has_track --n_video 50 --pan_im_json_file data/cityscapes_ext/panoptic_im_test_city_vps.json

python tools/eval_vpq.py \
  --submit_dir work_dirs/cityscapes_ext/ups_pano_vps_fusetrack_vpct/val0615_pans_unified/ \
  --truth_dir data/cityscapes_ext/validation/panoptic_video/ \
  --pan_gt_json_file data/cityscapes_ext/validation/panoptic_ann_val_city_vps.json
```
Files containing the predicted results will be generated as `pred.json` and `pan/*.png` at  `work_dirs/cityscapes_ext/ups_pano_ext_fusetrack_vpct/val_pans_unified/`. 

c. Cityscapes-VPS `test` split currently only allows evaluation on the codalab server. Please upload `submission.zip` to codalab server to see actual performances.
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
a. Train Fuse model on image-level Cityscapes.
```
bash ./tools/dist_train.sh configs/cityscapes/ups_pano_flow_tcea.py ${GPU_NUM}
```
b. Train FuseTrack model on video-level Cityscapes-VPS.
```
bash ./tools/dist_train.sh configs/cityscapes/ups_pano_ext_track_flow_tcea.py ${GPU_NUM}
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
