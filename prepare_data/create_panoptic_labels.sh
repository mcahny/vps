ROOT_DIR=$1

python prepare_data/create_panoptic_labels.py --root_dir $ROOT_DIR  --mode train
python prepare_data/create_panoptic_labels.py --root_dir $ROOT_DIR --mode val 

python prepare_data/create_panoptic_video_labels.py --root_dir $ROOT_DIR --mode train
python prepare_data/create_panoptic_video_labels.py --root_dir $ROOT_DIR --mode val


