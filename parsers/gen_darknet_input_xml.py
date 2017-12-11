import xml.etree.ElementTree as ET
import pickle
import os
from shutil import copyfile
import cv2
import uuid
import sys
sys.path.append('/home/bjxchang/dataset/')
from parsers import *

curr_pwd = os.getcwd()
output_root = curr_pwd

output_image_dir = output_root + '/images'
output_label_dir = output_root + '/labels'
output_patch_dir = output_root + '/patch'

is_view_all_labels = False
is_output_image_labeled =False
is_height_crop = False
view_country = 'India'
classes = []

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

if __name__ == "__main__":

    if os.path.exists(output_image_dir):
        os.system('rm -rf ' + output_image_dir)
    os.system('mkdir -p ' + output_image_dir)
    if os.path.exists(output_label_dir):
        os.system('rm -rf ' + output_label_dir)
    os.system('mkdir -p ' + output_label_dir)
    if os.path.exists(output_patch_dir):
        os.system('rm -rf ' + output_patch_dir)
    os.system('mkdir -p ' + output_patch_dir)

    objects_num = {}
    patch_classes = []
    list_file = open(os.path.join(output_root, view_country+'_obj.txt'), 'w')
    all_files = open(os.path.join(output_root,view_country+'_all.txt'),'w')
    class_file = open(os.path.join(output_patch_dir, view_country+'_obj_class.txt'), 'w')

    #specify the with and height of the image
    w = 1280
    h = 720
    height_crop = h
    if is_height_crop == True:
        height_crop = 448

    max_height = 0

#    subdir = []
#    f = open('./train_class.txt', 'r')
#    lines = f.readlines()
#    subdir = [x.strip() for x in lines]
#    print subdir
#    subdir = ['360(57)-2']
#    print subdir
    signs = ['car', 'autorickshaw', 'person', 'bike', 'truck']
    subdir = ['car_person_autorickshaw_20171110041857','car_person_autorickshaw_20171110044204','daylight']
    for dir in subdir:
        image_dir = os.path.join(curr_pwd,dir)
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.jpg'):
                    image_path = os.path.join(image_dir,file)
                    label_path = image_path.replace('.jpg','.xml')
                    print label_path
                    if not os.path.exists(label_path):
                        print '{} doesn\'t exists, will ignore.'.format(label_path)
                        continue
                    all_files.write(image_path+'\n')
                    objs = parse_xml(label_path)

                    bboxes = []
                    contain_interest_flag = False
                    img = cv2.imread(image_path)
                    for obj in objs.values()[0]:
                        if is_view_all_labels is not True:
                            print obj['name']
                            if obj['name'] not in signs:
                                continue
                        if obj['name'] not in classes:
                            classes.append(obj['name'])
                        detect_id = str(signs.index(obj['name']))
                        print obj['name']
                        print detect_id
                        b = (obj['bbox'][0],obj['bbox'][0]+obj['bbox'][2],
                                obj['bbox'][1],
                                obj['bbox'][1]+obj['bbox'][3])
                        if b[3] > height_crop:
                            continue
                        if b[3] > max_height:
                            max_height = b[3]
                        contain_interest_flag = True
                        bboxes.append([detect_id,b])
                        patch = img[b[2]:b[3], b[0]:b[1]]
                        patch_name = obj['name']+'-'+str(b[1]-b[0])+'x'+str(b[3]-b[2])+ \
                        '-'+str(uuid.uuid1())+'.jpg'
                        if obj['name'] not in patch_classes:
                            patch_classes.append(obj['name'])
                            os.system('mkdir -p '+os.path.join(output_patch_dir,obj['name']))
                        cv2.imwrite(os.path.join(os.path.join(output_patch_dir,obj['name']),patch_name),patch)
                        #cls_id = len(speedlimit)
                        #if obj['name'] in speedlimit:
                        #    cls_id = speedlimit.index(obj['name'])
                        cls_id = detect_id
                        class_file.write('/'+obj['name']+'/'+patch_name+' '+str(cls_id)+'\n')
                        if objects_num.has_key(obj['name']) is not True:
                            objects_num[obj['name']] = 0
                        objects_num[obj['name']] += 1

                    if contain_interest_flag is True:
                        label_str = ''
                        img = img[0:height_crop, :]
                        for box in bboxes:
                            bb = convert((w,height_crop), box[1])
                            label_str += str(box[0]) + " " + " ".join([str(a) for a in bb]) + '\n'
                            if is_output_image_labeled is True:
                                xmin = box[1][0]
                                xmax = box[1][1]
                                ymin = box[1][2]
                                ymax = box[1][3]
                                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
                                cv2.putText(img,box[0],(xmin+5,ymin-5), cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),2)
                        output_image_file_path = os.path.join(output_image_dir,file)
                        cv2.imwrite(output_image_file_path, img)

            # add the image to the list
                        list_file.write(output_image_file_path + '\n')

            # generate label file
                        output_label_file_path = os.path.join(output_label_dir, file.replace('.jpg', '.txt'))
                        output_label_file = open(output_label_file_path, 'w')
                        output_label_file.write(label_str)
                        output_label_file.close()

    print "max_height: %.2f"%(max_height)
    all_files.close()
    list_file.close()
    class_file.close()
    print objects_num
    print classes
