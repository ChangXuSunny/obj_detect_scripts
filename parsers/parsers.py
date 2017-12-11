import xml.etree.ElementTree as ET
import pickle
import os
import json


def parse_json(filename):
    try:
        data = json.loads(open(filename).read())
    except:
        raise Exception ('can not parse the json file:{}'.format(filename))

    labelinfo = {}
    labelnum = 0
    for frame in data["Frames"]:
        boxes_info = []
        for box in frame["LabelingRects"]:
            labelnum +=1
            obj_struct = {}
            obj_struct['name'] = 'unlabel'
            obj_struct['category'] = 'unlabel'
            if "Attributes" in box.keys() and box['Attributes'] is not None:
                obj_struct['category'] = box['Attributes']['TrafficSignCategory']
                obj_struct['name'] = box['Attributes']['TrafficSignMeaning']
            obj_struct['pose'] = None
            obj_struct['truncated'] = None
            obj_struct['difficult'] = False
            obj_struct['bbox'] = [int(box["X"]),
                int(box["Y"]),
                int(box["Width"]),
                int(box["Height"])]
            boxes_info.append(obj_struct)

        if len(boxes_info) is not 0:
            labelinfo[str(frame['frameIndex'])] = boxes_info
    print "#labels: "+str(labelnum)
    return labelinfo


def parse_xml(filename):
    if not os.path.exists(filename):
        raise Exception ('can not parse the json file:{}'.format(filename))
    tree = ET.parse(filename)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    labelinfo = {}
    boxes_info = []
    for obj in root.iter('object'):
        obj_struct = {}
        obj_struct['category'] = 'others'
        obj_struct['name'] = obj.find('name').text
        if obj_struct['name'].find('red-sl') != -1:
            obj_struct['category'] = 'SpeedLimit'
            obj_struct['name'] = obj_struct['name'][6:]
        obj_struct['pose'] = None
        obj_struct['truncated'] = None
        obj_struct['difficult'] = False
        xmlbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(xmlbox.find('xmin').text),
                int(xmlbox.find('ymin').text),
                int(xmlbox.find('xmax').text) - int(xmlbox.find('xmin').text),
                int(xmlbox.find('ymax').text) - int(xmlbox.find('ymin').text)]
        boxes_info.append(obj_struct)

    if len(boxes_info) is not 0:
        keyname = filename.split('/')[-1].replace('.xml','.jpg')
        labelinfo[keyname] = boxes_info
    print "#labels: "+str(len(boxes_info))
    print labelinfo
    return labelinfo

