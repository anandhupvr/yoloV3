from utils.Bbox import Bbox
import numpy as np
import os
import glob
import cv2
from PIL import Image



def convert_to_bbox(labels_all):
    objs = []
    for label in labels_all:
        c, x, y, w, h = label.split(" ")
        objs.append(Bbox(float(x), float(y), float(w), float(h), int(c)))
    return objs

def compute_iou(box1, box2):

    intersect_w = _interval_overlap([box1.x_ax, box1.y_ax], [box2.x_ax, box2.y_ax])
    intersect_h = _interval_overlap([box1.w_ax, box1.h_ax], [box2.w_ax, box2.h_ax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.y_ax-box1.x_ax, box1.h_ax-box1.w_ax
    w2, h2 = box2.y_ax-box2.x_ax, box2.h_ax-box2.w_ax
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3


def train_test_split(data_path, *args):
    if len(args) == 0:
        percentage_test = 10
    else:
        percentage_test = args[0]
    img_count = len(os.listdir(data_path+"images"))
    file_train = open(data_path + 'train.txt', 'w+')  
    file_test = open(data_path + 'test.txt', 'w+')
    counter = 0
    index_test = round(img_count * percentage_test / 100)
    all_items = glob.glob(data_path+"images/*.jpg")
    for i in all_items:
        if counter < index_test:
            file_test.write(i+"\n")
            counter += 1
        else:
            file_train.write(i+"\n")


def manip_image_and_label(config, image_file, objs):
    test = image_file
    image = cv2.imread(test.strip("\n"))
    h_, w_, c = image.shape
    image = cv2.resize(image, (config["IMAGE_H"], config["IMAGE_W"]))
    for obj in objs:
        # converting yolo to bbox
        x_norm, w_norm, y_norm, h_norm = obj.x_ax, obj.y_ax, obj.w_ax, obj.h_ax
        x_mid = x_norm * w_
        y_mid = y_norm * h_
        w = w_ * w_norm
        h = h_ * h_norm
        obj.x_ax = (x_mid - w / 2)
        obj.y_ax = (y_mid - h / 2)
        obj.w_ax = w
        obj.h_ax = h

        # manipulating label relative to resized image
        obj.x_ax = int(obj.x_ax * float(config['IMAGE_W']) / w_)
        obj.x_ax = max(min(obj.x_ax, config['IMAGE_W']), 0)
        obj.w_ax = int(obj.w_ax * float(config['IMAGE_W']) / w_)
        obj.w_ax = max(min(obj.w_ax, config['IMAGE_W']), 0)
        obj.y_ax = int(obj.y_ax * float(config['IMAGE_H']) / h_)
        obj.y_ax = max(min(obj.y_ax, config['IMAGE_H']), 0)
        obj.h_ax = int(obj.h_ax * float(config['IMAGE_H']) / h_)
        obj.h_ax = max(min(obj.h_ax, config['IMAGE_H']), 0)

    return image, objs


class weight_reader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file)

    def read_bytes(self, size):
        self.offset = self.offset + size
        return all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4
