mkdir -p work_dirs/flownet
mkdir -p work_dirs/cityscapes_vps/fusetrack_vpct
mkdir -p work_dirs/cityscapes/fuse_vpct
mkdir -p work_dirs/viper/fuse

# download trained weight
cd work_dirs/flownet
gdown https://drive.google.com/uc?id=1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da

cd ../cityscapes_vps/fusetrack_vpct/
gdown https://drive.google.com/uc?id=1KcHYnghbs2KC6hQc7QVkPkEiJMrLr73s
mv cityscapes_vps_fusetrack_latest.pth latest.pth

cd ../../cityscapes/fuse_vpct
gdown https://drive.google.com/uc?id=1t69I1u0QKl-N4eciYv3UYXFOQSYb25cD
mv cityscapes_fuse_latest.pth latest.pth

cd ../../viper/fuse
gdown https://drive.google.com/uc?id=1_4Np8-rBGHchL2nU1sOjK3mwnLX9asZC
mv viper_fuse_latest.pth latest.pth

