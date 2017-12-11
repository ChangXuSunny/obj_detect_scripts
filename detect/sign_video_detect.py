import caffe
import os
import numpy as np
import sys, getopt
import cv2
import json
import time
import optparse
import ConfigParser
from yolodetector import ObjectDetector

video_path = '/root/msil/dataset/worst_drivers_and_traffic_of_Bangalore.mp4'
out_video_path = 'out.avi'
det_results_path = 'det_results.txt'

extract_dir = 'extract/'
output_dir = 'dets/'

tmpfile = '/tmp/tmp.jpg'

detect_model = 'models/tiny-yolo-kitti-640-relu-512.prototxt'
detect_weights = 'models/tiny-yolo-kitti-640-relu-512.caffemodel'

classify_model = './lenet_deploy.prototxt'
classify_weight = './lenet_iter_20000.caffemodel'

if os.path.exists(output_dir):
    os.system('rm -rf ' + output_dir)
os.system('mkdir -p ' + output_dir)

if os.path.exists(extract_dir):
    os.system('rm -rf ' + extract_dir)
os.system('mkdir -p ' + extract_dir)

yolo_detector = ObjectDetector(detect_model, detect_weights, False)
#classifier = SignClassifier(classify_model, classify_weight, False)

cap = cv2.VideoCapture(video_path)
if (cap.isOpened()== False):
    print("Error opening video stream or file")

fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
out = cv2.VideoWriter(out_video_path, fourcc, cap.get(cv2.cv.CV_CAP_PROP_FPS), (1280, 720))
#out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'XVID'),30, (1280,720))


det_fd = open(det_results_path, 'w')

frame_idx = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is not True:
        break;
    print 'Processing frame: {}'.format(frame_idx)
    image_file = os.path.join(extract_dir, '%06d.jpg'%frame_idx)
    cv2.imwrite(image_file, frame)
    results = yolo_detector.detect(image_file)
    print results
    img = cv2.imread(image_file)
    for result in results:
        xmin = result['xmin']
        ymin = result['ymin']
        xmax = result['xmax']
        ymax = result['ymax']
        label = result['label']
        confidence = result['confidence']

        det_fd.write('%06d %s %.2f %d %d %d %d\n'%(frame_idx, label, confidence, xmin, ymin, xmax-xmin, ymax-ymin))

        cv2.imwrite(tmpfile, img[ymin:ymax+1, xmin:xmax+1])
#        classify_result = classifier.detect(tmpfile)
#        print classify_result
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        cv2.rectangle(img,(xmin-60,ymin-17),(xmax+60,ymin),(0,255,0),-1)
        cv2.putText(img, \
                    '%s :%.2f[%dx%d]'%(label, confidence, xmax-xmin, ymax-ymin),\
                    (xmin-57,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)

    cv2.imwrite(os.path.join(output_dir, '%06d.jpg'%frame_idx), img)
    out.write(img)

    frame_idx = frame_idx + 1

out.release()
det_fd.close()

