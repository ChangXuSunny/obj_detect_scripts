import os
import numpy as np
import sys, getopt
import cv2
import json
import time
import optparse
import ConfigParser

jpg_path = './trcks'
out_video_path = '2017-05-08-194848_ver4101.avi'


fourcc = cv2.cv.CV_FOURCC('D', 'I', 'V', 'X')
out = cv2.VideoWriter(out_video_path, fourcc, 30, (1280, 720))

image_list = [int(x.split('.')[-2]) for x in os.listdir(jpg_path) if x.endswith('.jpg')]
sorted_ind = np.argsort(image_list)
image_list = ['2017-05-08-194848_ver4101.mp4.'+str(image_list[x])+'.jpg' for x in sorted_ind]

for image in image_list:
    print image
    img = cv2.imread(os.path.join(jpg_path,image))
    out.write(img)

out.release()

