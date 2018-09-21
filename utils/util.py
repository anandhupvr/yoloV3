from utils.Bbox import Bbox
import numpy as np


def convert_to_bbox(labels_all):
    objs = []
    for label in labels_all:
        c, x, y, w, h = label.split(" ")
        objs.append(Bbox(float(x), float(y), float(w), float(h), int(c)))
    return objs


class weight_reader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file)

    def read_bytes(self, size):
        self.offset = self.offset + size
        return all_weights[self.offset - size:self.offset]

    def reset(self):
        self.offset = 4
