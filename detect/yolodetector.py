import numpy as np
import cv2
import sys
import caffe
import math
from utils import sigmoid, softmax, overlap, cal_iou, apply_nms
from data import mat_dump_int
label_name = {1: "autorickshaw", 3: "bike", 0:"car", 2:"person", 4:"truck"}
biases = [2.136601,1.803864, 4.965101,3.824877, 7.378543,6.567224, 1.010301,1.148794, 3.693888,2.338183]


detector_laye_name = 'conv9'
nms_thresh = 0.45
class ObjectDetector(caffe.Net):
    def __init__(self, model_file, pretrained_file, gpu_mode, thresh=0.4):
        if gpu_mode:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        #self.transformer.set_channel_swap('data', (2,1,0))
        self.thresh = thresh

    def detect(self, img_filename, crop=None):
        image = caffe.io.load_image(img_filename)
        if crop is not None:
            image = image[crop[1]:crop[1]+crop[3],crop[0]:crop[0]+crop[2],:]
        image_w = image.shape[1]
        image_h = image.shape[0]

        resized_image = self.transformer.preprocess('data', image)
        #mat_dump_int('pred.txt', , 16, 9)
        out = self.forward_all(data=np.asarray([self.transformer.preprocess('data', image)]))
        res = out[detector_laye_name][0]

        fw = res.shape[2]
        fh = res.shape[1]
        anchor_num = len(biases) / 2
        res = res.transpose(1,2,0).reshape((fw * fh, anchor_num, res.shape[0]/anchor_num))
        #mat_dump_int('pred.txt', res, 16, 9)

        classes_num = len(label_name)
        boxes = list()
        for h in range(fh):
            for w in range(fw):
                for n in range(anchor_num):
                    x = (w + sigmoid(res[h*fw+w][n][0])) / float(fw)
                    y = (h + sigmoid(res[h*fw+w][n][1])) / float(fh)
                    ww = (math.exp(res[h*fw+w][n][2])*biases[2*n]) / float(fw)
                    hh = (math.exp(res[h*fw+w][n][3])*biases[2*n+1]) / float(fh)
                    obj_score = sigmoid(res[h*fw+w][n][4])
                    cls_probs = softmax(res[h*fw+w][n][5:5+classes_num])

                    if obj_score * max(cls_probs) > self.thresh:
                        boxes.append([x, y, ww, hh, np.argmax(cls_probs), obj_score * max(cls_probs)])
                        #print x, y, ww, hh, np.argmax(cls_probs), obj_score * max(cls_probs)

        res = apply_nms(boxes, nms_thresh)

        objects_list = []
        for box in res:
            name = label_name[box[4]]
            prob = box[5]
            xmin = max(0, (box[0]-box[2]/2.0) * image_w)
            xmax = min(image_w, (box[0]+box[2]/2.0) * image_w)
            ymin = max(0, (box[1]-box[3]/2.0) * image_h)
            ymax = min(image_h, (box[1]+box[3]/2.0) * image_h)

            if prob > self.thresh:
                object_item = {}
                object_item['label'] = name
                object_item['confidence'] = prob
                object_item['xmin'] = int(round(xmin))
                object_item['ymin'] = int(round(ymin))
                object_item['xmax'] = int(round(xmax))
                object_item['ymax'] = int(round(ymax))
                objects_list.append(object_item)
        return objects_list

if __name__  == '__main__':

    model_def = './models/tiny-yolo-sl-1280-f16.prototxt'
    model_weights = './models/tiny-yolo-sl-1280-f16.caffemodel'

    image_name = './images/kitti-2.png'

    detector = ObjectDetector(model_def, model_weights, True)
    results = detector.detect(image_name)
    print results

    img = cv2.imread(image_name)
    for result in results:
        xmin = result['xmin']
        ymin = result['ymin']
        xmax = result['xmax']
        ymax = result['ymax']
        label = result['label']
        confidence = result['confidence']
        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
        cv2.rectangle(img,(xmin,ymin-20),(xmax+30,ymin),(125,125,125),-1)
        cv2.putText(img,label + ':%.2f' % confidence,(xmin+5,ymin-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)

    cv2.imshow('YOLO detection',img)
    cv2.waitKey(1000000)
