ROOT_DIR=$1
SRC_IM=$2
SRC_GT=$3

python prepare_data/fetch_city_images.py --src_dir $SRC_IM --dst_dir $ROOT_DIR --mode train
python prepare_data/fetch_city_images.py --src_dir $SRC_IM --dst_dir $ROOT_DIR --mode val
python prepare_data/fetch_city_images.py --src_dir $SRC_IM --dst_dir $ROOT_DIR --mode test

python prepare_data/merge_datasets.py --src_dir $SRC_GT --dst_dir $ROOT_DIR --mode train
python prepare_data/merge_datasets.py --src_dir $SRC_GT --dst_dir $ROOT_DIR --mode val



