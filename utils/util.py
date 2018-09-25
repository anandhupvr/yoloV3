from utils.Bbox import Bbox
import numpy as np


def convert_to_bbox(labels_all):
    objs = []
    for label in labels_all:
        c, x, y, w, h = label.split(" ")
        objs.append(Bbox(float(x), float(y), float(w), float(h), int(c)))
    return objs


# class weight_reader:
#     def __init__(self, weight_file):
#         self.offset = 4
#         self.all_weights = np.fromfile(weight_file)

#     def read_bytes(self, size):
#         self.offset = self.offset + size
#         return all_weights[self.offset - size:self.offset]

#     def reset(self):
#         self.offset = 4

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