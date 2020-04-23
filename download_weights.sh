mkdir -p work_dirs/flownet
mkdir -p work_dirs/cityscapes/ups_pano_flow_tcea_vp
mkdir -p work_dirs/viper/ups_pano_flow_tcea
# download trained weight
cd work_dirs/cityscapes/ups_pano_flow_tcea_vp
gdown https://drive.google.com/uc?id=1t69I1u0QKl-N4eciYv3UYXFOQSYb25cD
mv cityscapes_fuse_latest.pth latest.pth

cd ../../viper/ups_pano_flow_tcea
gdown https://drive.google.com/uc?id=1_4Np8-rBGHchL2nU1sOjK3mwnLX9asZC
mv viper_fuse_latest.pth latest.pth