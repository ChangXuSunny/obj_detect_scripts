rm -rf ./backup
mkdir backup
nohup ./darknet detector train ./india.data ./tiny-yolo-kitti-640-relu-512.cfg ./darknet.weights.conv10 &
