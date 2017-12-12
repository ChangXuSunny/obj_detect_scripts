import math

def sigmoid(x):
    return 1.0 / (1 + math.exp(-x * 1.0))

def softmax(cls):
    large = max(cls);
    for i in range(len(cls)):
        cls[i] = math.exp(cls[i] - large);
    s = sum(cls);
    for i in range(len(cls)):
        cls[i] = cls[i] * 1.0 / s;
    return cls

def overlap(x1, w1, x2, w2): #x1 ,x2 are two box center x
    left = max(x1 - w1 / 2.0, x2 - w2 / 2.0)
    right = min(x1 + w1 / 2.0, x2 + w2 / 2.0)
    return right - left

def cal_iou(box, truth):
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if w < 0 or h < 0:
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    return inter_area * 1.0 / union_area

def apply_nms(boxes, thres):
    sorted_boxes = sorted(boxes,key=lambda d: d[5])[::-1]
    p = dict()
    for i in range(len(sorted_boxes)):
        if i in p:
            continue
        truth =  sorted_boxes[i]
        for j in range(i+1, len(sorted_boxes)):
            if j in p:
                continue
            box = sorted_boxes[j]
            iou = cal_iou(box, truth)
            if iou >= thres:
                p[j] = 1
    res = list()
    for i in range(len(sorted_boxes)):
        if i not in p:
            res.append(sorted_boxes[i])
    return res
