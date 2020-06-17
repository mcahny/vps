ROOT_DIR=$1
MODE=$2
SRC_GT=$3
SRC_IM=$4

python prepare_city_vps/merge_datasets.py --mode $MODE --src_dir $SRC_GT --dst_dir $ROOT_DIR
python prepare_city_vps/fetch_city_images.py --mode $MODE --src_dir $SRC_IM --dst_dir $ROOT_DIR
# cd prepare_city_vps
# pip install .
# cd ..
python prepare_city_vps/create_panoptic_labels.py --mode $MODE --root_dir $ROOT_DIR
python prepare_city_vps/create_panoptic_video_labels.py --mode $MODE --root_dir $ROOT_DIR