import os
import numpy as np
from utils.Bbox import Bbox
from utils import util
import cv2
from PIL import Image


def get_data(config, root_dir='/dataset/'):

    x_batch = np.zeros(
        [config["BATCH_SIZE"],
         config["IMAGE_W"],
         config["IMAGE_H"],
         3], np.float32)
    anchors = [Bbox(0, 0, config["ANCHORS"][2 * i], config["ANCHORS"][2 * i + 1])
               for i in range(int((len(config["ANCHORS"])) / 2))]
    y_batch = np.zeros(
        [config["BATCH_SIZE"],
         config["GRID_H"],
         config["BOX"],
         4 + 1 + config["CLASS"]],
        np.float32)

    for file in os.listdir(os.getcwd() + root_dir + "images/"):
        labels_file = open(os.getcwd() + "/dataset/labels/" +
                           file[:-3] + "txt").readlines()
        labels_all = [labels_file[i] for i, l in enumerate(labels_file)]
        objs = util.convert_to_bbox(labels_all)
        image, objs = manip_image_and_label(
            os.getcwd() + root_dir + "images/" + file, objs, config)
        for obj in objs:
            class_vector = np.zeros(config["CLASS"])
            class_vector[obj.cat] = 1
            center_x, center_y, center_w, center_h = obj.x_ax, obj.y_ax, obj.w_ax, obj.h_ax
            center_x = center_x / (config["IMAGE_W"]/config["GRID_W"])
            center_y = center_y / (config["IMAGE_H"]/config["GRID_H"])

            center_w = center_w / (config["IMAGE_W"]/config["GRID_W"])
            center_h = center_h / (config["IMAGE_H"]/config["GRID_H"])


            print ("centerx = {} centery = {} centerw = {} centerh = {}".format(center_x, center_y, center_w, center_h))

def manip_image_and_label(image_file, objs, config):
    image = cv2.imread(image_file)
    h_, w_, c = image.shape
    print(w_, h_, c)
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
