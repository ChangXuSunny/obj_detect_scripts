import os
import numpy as np
import sys, getopt
import cv2
import json
import time
import optparse
import ConfigParser
from yolodetector import ObjectDetector
from sign_classifier import SignClassifier
sys.path.append('/home/bjxchang/xiaomi/detection/3Dloc')
from image23D import *
from sort import *

if len(sys.argv) < 2:
    print "Usage: python sign_track.py <output_file name>"
    sys.exit()

name_prefix = sys.argv[1]
print name_prefix

images_dir = '/home/bjxchang/dataset/China/d_226/images'
#images_dir = '/home/bjxchang/dataset/China/video2/2017-05-08-194848_ver4101.mp4-jpg/'
output_dir = './'+name_prefix+'_dets/'

tmpfile = '/tmp/tmp.jpg'

detect_model = 'models/tiny-yolo-sl-1280-f16.prototxt'
detect_weights = 'models/tiny-yolo-sl-1280-f16.caffemodel'

classify_model = 'models/lenet_deploy.prototxt'
classify_weight = 'models/lenet_iter_20000.caffemodel'

thresh=0.05
Sc = 3.14*300*300
img_width = 1280
img_height = 720
class_filter = False
track_flag = False
sort_flag = False

if os.path.exists(output_dir):
    os.system('rm -rf ' + output_dir)
os.system('mkdir -p ' + output_dir)

yolo_detector = ObjectDetector(detect_model, detect_weights, False, thresh)
classifier = SignClassifier(classify_model, classify_weight, False)
mot_tracker = Sort(2,3)

det_fd = open(name_prefix+'_detect_'+str(thresh)+'.txt', 'w')

image_list = [x for x in os.listdir(images_dir) if x.endswith('.jpg')]
if sort_flag is True:
    image_list = [int(x.split('.')[-2]) for x in os.listdir(images_dir) if x.endswith('.jpg')]
    sorted_ind = np.argsort(image_list)
    image_list = ['2017-05-08-194848_ver4101.mp4.'+str(image_list[x])+'.jpg' for x in sorted_ind]



classes = ["sl5", "sl20", "sl30", "sl40", "sl50", "sl60", "sl80", "sl100", "noise", "error"]

for image_name in image_list:
    image_file = os.path.join(images_dir,image_name)
    img = cv2.imread(image_file)
    print image_file

    results = yolo_detector.detect(image_file, (0, 0, 1280, 448))
    dets = []
    for result in results:
        cv2.imwrite(tmpfile, img[result['ymin']:result['ymax']+1, result['xmin']:result['xmax']+1])
        classify_result = classifier.detect(tmpfile)

        if class_filter is True:
            if classify_result['cls'] == 8 or classify_result['confidence'] < 0.5:
                continue
        tmp_det = [int(result['xmin']), int(result['ymin']), int(result['xmax']), int(result['ymax']), 0, result['confidence'], \
                classify_result['cls'], classify_result['confidence']]
        dets.append(tmp_det)
    dets = np.asarray(dets)
    print "original detections"
    print dets

    if track_flag is True:
        trackers = mot_tracker.update(dets)
        dets = trackers

    print "output detections"
    print dets
    for result in dets:
        xmin = int(result[0])
        ymin = int(result[1])
        xmax = int(result[2])
        ymax = int(result[3])
        label = result[4]
        detect_confidence = result[5]
        classId = int(result[6])
        class_prob = result[7]


        u = img_width-(xmin+xmax)/2
        v = img_height-(ymin+ymax)/2
        iw = xmax - xmin
        ih = ymax - ymin
        cor = imageto3D(u,v,iw,ih,Sc)
        c_x = cor[0][0]/1000
        c_y = cor[0][1]/1000
        c_z = cor[0][2]/1000


        det_fd.write('%s, %s, %.2f, %d, %d, %d, %d, %s, %.2f, %.2f, %.2f, %.2f\n'%(image_name, str(label), detect_confidence, xmin, ymin, xmax-xmin, ymax-ymin, classes[classId], class_prob,c_x,c_y,c_z))

        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        #if (xmax-xmin)*(ymax-ymin) > 30*30:
        if True:
            cv2.putText(img, classes[classId] + \
                    ':%.2f,%.2f[%dx%d](%.2f,%.2f,%.2f)'%(detect_confidence, class_prob, xmax-xmin, ymax-ymin,c_x,c_y,c_z),\
                    (xmin-57,ymin-5),cv2.FONT_HERSHEY_SIMPLEX,1,(255,192,203),2)

    cv2.imwrite(os.path.join(output_dir, image_name), img)
det_fd.close()
