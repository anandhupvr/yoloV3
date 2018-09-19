from utils.Bbox import Bbox
import numpy as np

def convert_to_bbox(labels_all):
	objs = []
	for label in labels_all:
		c, x, y, w, h = label.split(" ")
		objs.append(Bbox(float(x), float(y), float(w), float(h), int(c)))
	return objs
