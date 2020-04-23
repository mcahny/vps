# install UPSNet dependencies
cd mmdet/models/utils/upsnet/bbox; python setup.py build_ext --inplace
cd ../nms; python setup.py build_ext --inplace