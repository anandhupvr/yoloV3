import os
import numpy as np
from utils.Bbox import Bbox
from utils import util
import cv2


class DataLoader:

    def __init__(self, data_path, config):
        self.batch_ptr = 0
        self.config = config
        self.data_path = data_path
        self.anchors = [Bbox(0, 0, self.config["ANCHORS"][2 * i], self.config["ANCHORS"][2 * i + 1])
                         for i in range(int((len(self.config["ANCHORS"])) / 2))]
        util.train_test_split(self.data_path)

    def next_batch(self, batch_size):
        instance_count = 0
        max_iou = -1
        best_prior = -1
        ptr = self.batch_ptr * batch_size

        x_batch = np.zeros(
            [batch_size,
             self.config["IMAGE_W"],
             self.config["IMAGE_H"],
             3], np.float32)
        anchors = [Bbox(0, 0, self.config["ANCHORS"][2 * i], self.config["ANCHORS"][2 * i + 1])
                   for i in range(int((len(self.config["ANCHORS"])) / 2))]

        y_batch = np.zeros(
            [batch_size,
             self.config["GRID_W"],
             self.config["GRID_H"],
             self.config["BOX"],
             4 + 1 + self.config["CLASS"]],
            np.float32)

        image_files = open(os.path.join(
                                        self.data_path,
                                        "train.txt"),
                                         "r").readlines()[ptr:ptr+batch_size]

        for file in image_files:
            labels_file = open((file.split("images")[0]+"labels"+file.split("images")[1].replace("jpg", "txt")).strip("\n")).readlines()
            labels_all = [labels_file[i] for i, l in enumerate(labels_file)]
            objs = util.convert_to_bbox(labels_all)
            image, objs = util.manip_image_and_label(self.config, file, objs)
            for obj in objs:
                class_vector = np.zeros(self.config["CLASS"])
                class_vector[obj.cat] = 1
                center_x, center_y, center_w, center_h = obj.x_ax, obj.y_ax, obj.w_ax, obj.h_ax
                center_x = center_x / (self.config["IMAGE_W"]/self.config["GRID_W"])
                center_y = center_y / (self.config["IMAGE_H"]/self.config["GRID_H"])

                center_w = center_w / (self.config["IMAGE_W"]/self.config["GRID_W"])
                center_h = center_h / (self.config["IMAGE_H"]/self.config["GRID_H"])
                # print ("centerx = {} centery = {} centerw = {} centerh = {}".format(center_x, center_y, center_w, center_h))
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
                bbox = [center_x, center_y, center_w, center_h]
                box = Bbox(0, 0, center_w, center_h)
                for i in range(len(anchors)):
                    iou = util.compute_iou(anchors[i], box)

                    if iou > max_iou:
                        max_iou = iou
                        best_prior = i

                y_batch[instance_count, grid_x, grid_y, best_prior, 0:4] = bbox
                y_batch[instance_count, grid_x, grid_y, best_prior, 4] = 1
                y_batch[instance_count, grid_x, grid_y, best_prior, 5:5+self.config['CLASS']] = class_vector
                x_batch[instance_count] = image

            instance_count += 1
        return x_batch, y_batch

