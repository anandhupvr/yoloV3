import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from xml.etree import ElementTree as ET

import torchvision.transforms as transforms

from utils.util import iou, nms, cells_to_bboxes, plot_image


class CocoDataset(Dataset):
    def __init__(self, data_dir, annotation_file, anchors,
                 image_size=416, grid_sizes=[13, 26, 52],
                 num_classes=80):
        
        # self.data_frame = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.annotation_file = annotation_file

        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.grid_sizes = grid_sizes
        self.image_size = image_size
        self.num_classes = num_classes

        self.number_of_anchor_per_scale = 3
        self.ignore_iou_thresh = 0.5
        self._prepare_dataset()

        self.transforms = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def __len__(self):
        return len(self.image_list)

    def _load_label(self, label_path):
        label_path = label_path.replace('images', 'labels').replace('jpg', 'txt')
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        print(bboxes)
        return bboxes
    
    def _load_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    def _prepare_dataset(self):
        with open(os.path.join(self.data_dir, self.annotation_file), 'r') as f:
            self.image_list = f.read().splitlines()
        # self.images = [self._load_img(os.path.join(self.data_dir, img_path)) for img_path in self.image_list]



    def __getitem__(self, idx):
        path = os.path.normpath(os.path.join(self.data_dir, self.image_list[idx]))
        image = self._load_image(path)
        bboxes = self._load_label(path)
        if True:
            # image = self.transforms(image)

            # size_w, size_h = image.size
            # scale_x, scale_y = 416/size_w, 416/size_h
            image = self.transforms(image)
            # for box in bboxes:
            #     box[0] *= scale_x
            #     box[1] *= scale_y
            #     box[2] *= scale_x
            #     box[3] *= scale_y

        targets = [torch.zeros((3, S, S, 5+self.num_classes)) for S in self.grid_sizes]
        for box in bboxes:
            # iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, False)
            anchor_indices = iou_anchors.argsort(dim=0, descending=True)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.number_of_anchor_per_scale
                anchor_on_scale = anchor_idx % self.number_of_anchor_per_scale
                S = self.grid_sizes[scale_idx]
                i, j = int(S * y), int(S * x) # Find cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # [0, 1]
                    width_cell, height_cell = (width * S, height * S)
                    box_coordinates = torch.tensor( [x_cell, y_cell, width_cell, height_cell] )

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore prediction


        return np.array(image), tuple(targets)


class VocDataset(Dataset):
    def __init__(self, data_dir, num_classes, anchors, image_set='train', image_size=416, grid_sizes=[13, 26, 52]):

        self.data_dir = data_dir
        self.num_classes = num_classes
        self.image_set = image_set
        self.image_size = image_size
        self.grid_sizes = grid_sizes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.transforms = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.image_paths = []
        self.annotations = []
        self._prepare_dataset()
        self.number_of_anchor_per_scale = 3
        self.ignore_iou_thresh = 0.5

        self.VOC_CLASSES = [
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor']
        self.pascal_voc_2007_categories = { 
                                0: 'aeroplane', 1: 'bicycle', 2: 'bird', 3: 'boat',
                                4: 'bottle',5: 'bus',6: 'car',7: 'cat',8: 'chair',9: 'cow',
                                10: 'diningtable',11: 'dog',12: 'horse',13: 'motorbike',14: 'person',
                                15: 'pottedplant', 16: 'sheep', 17: 'sofa', 18: 'train', 19: 'tvmonitor'
                            }
        self.category_name_to_id = {v: k for k, v in self.pascal_voc_2007_categories.items()}


    def _prepare_dataset(self):
        
        image_set_file = os.path.join(self.data_dir, 'ImageSets', 'Main', f'{self.image_set}.txt')
        with open(image_set_file, 'r') as f:
            image_ids = f.read().splitlines()
        for image_name in image_ids:
            image_name = image_name.split(' ')[0]
            image_path = os.path.join(self.data_dir, 'JPEGImages', f'{image_name}.jpg')
            annotation_path = os.path.join(self.data_dir, 'Annotations', f'{image_name}.xml')

            self.image_paths.append(image_path)
            self.annotations.append(annotation_path)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotations[idx]
        image = Image.open(image_path).convert("RGB")
        bboxes = self._parse_annotation(annotation_path)
        if True:
            # size_w, size_h = image.size
            # scale_x, scale_y = 416/size_w, 416/size_h
            # for box in bboxes:
            #     box[0] *= scale_x
            #     box[1] *= scale_y
            #     box[2] *= scale_x
            #     box[3] *= scale_y
            image = self.transforms(image)
        targets = [torch.zeros((3, S, S, 5+self.num_classes)) for S in self.grid_sizes]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, False)
            anchor_indices = iou_anchors.argsort(dim=0, descending=True)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.number_of_anchor_per_scale # 0, 1, 2
                anchor_on_scale = anchor_idx % self.number_of_anchor_per_scale # 0, 1, 2
                S = self.grid_sizes[scale_idx]
                i, j = int(S * y), int(S * x) # Find cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # [0, 1]
                    width_cell, height_cell = (width * S, height * S)
                    box_coordinates = torch.tensor( [x_cell, y_cell, width_cell, height_cell] )

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore prediction

        return np.array(image), tuple(targets)


    def _parse_annotation(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)
        bboxes = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            category_id = self.category_name_to_id[class_name]
            
            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)
            width = x_max - x_min
            height = y_max - y_min
            
            x_min_norm = (x_min + (width/2)) / image_width
            y_min_norm = (y_min + (height/2)) / image_height
            width_norm = width / image_width
            height_norm = height / image_height

            bboxes.append([x_min_norm, y_min_norm, width_norm, height_norm, category_id])
        
        return np.array(bboxes)


class DogVsCatDataset(Dataset):
    def __init__(self, data_dir, num_classes, anchors, image_size=416, grid_sizes=[13, 26, 52]):

        self.data_dir = data_dir
        self.num_classes = num_classes
        self.image_size = image_size
        self.grid_sizes = grid_sizes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # for all 3 scales
        self.transforms = transforms.Compose([
            transforms.Resize((416, 416)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.image_paths = []
        self.annotations = []
        self._prepare_dataset()
        self.number_of_anchor_per_scale = 3
        self.ignore_iou_thresh = 0.5

        self.dogvscat = { 
                                'dog' : 0, 'cat' : 1
                            }


    def _prepare_dataset(self):
        
        image_set_file = os.path.join(self.data_dir, 'train.txt')
        with open(image_set_file, 'r') as f:
            lines = f.read().splitlines()
        for line in lines:
            image_name, label_name = line.split(' ')
            image_path = os.path.join(self.data_dir, 'files', image_name)
            annotation_path = os.path.join(self.data_dir, 'files', label_name)

            self.image_paths.append(image_path)
            self.annotations.append(annotation_path)

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotations[idx]
        image = Image.open(image_path).convert("RGB")
        bboxes = self._parse_annotation(annotation_path)
        if True:
            # size_w, size_h = image.size
            # scale_x, scale_y = 416/size_w, 416/size_h
            # for box in bboxes:
            #     box[0] *= scale_x
            #     box[1] *= scale_y
            #     box[2] *= scale_x
            #     box[3] *= scale_y
            image = self.transforms(image)
        targets = [torch.zeros((3, S, S, 5+self.num_classes)) for S in self.grid_sizes]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors, False)
            anchor_indices = iou_anchors.argsort(dim=0, descending=True)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.number_of_anchor_per_scale # 0, 1, 2
                anchor_on_scale = anchor_idx % self.number_of_anchor_per_scale # 0, 1, 2
                S = self.grid_sizes[scale_idx]
                i, j = int(S * y), int(S * x) # Find cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # [0, 1]
                    width_cell, height_cell = (width * S, height * S)
                    box_coordinates = torch.tensor( [x_cell, y_cell, width_cell, height_cell] )

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore prediction

        return image, tuple(targets)


    def _parse_annotation(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        image_width = int(root.find('size/width').text)
        image_height = int(root.find('size/height').text)
        bboxes = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            category_id = self.dogvscat[class_name]
            
            bbox = obj.find('bndbox')
            x_min = float(bbox.find('xmin').text)
            y_min = float(bbox.find('ymin').text)
            x_max = float(bbox.find('xmax').text)
            y_max = float(bbox.find('ymax').text)
            width = x_max - x_min
            height = y_max - y_min
            
            x_min_norm = (x_min + (width/2)) / image_width
            y_min_norm = (y_min + (height/2)) / image_height
            width_norm = width / image_width
            height_norm = height / image_height

            bboxes.append([x_min_norm, y_min_norm, width_norm, height_norm, category_id])
        
        return np.array(bboxes)

def test_dog_cat():
    data_dir = "/home/anandhu/Documents/works/yoloV3/data/dogVScat/"
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]  # Note these have been rescaled to be between [0, 1]

    dataset = DogVsCatDataset(data_dir, 2, ANCHORS)

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y, path in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]

        boxesn = nms(boxes, 1, 0.5)
        # print(boxes)
        image = Image.open(path[0]).convert("RGB")
        image = image.resize((416, 416), Image.Resampling.LANCZOS)
        # print(boxesn)
        plot_image(image, boxesn, 'voc')



def test_voc():
    data_dir = "/home/anandhu/Documents/works/yoloV3/data/VOCdevkit/VOC2007/"
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]  # Note these have been rescaled to be between [0, 1]

    dataset = VocDataset(data_dir, 20, ANCHORS)

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y, path in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]

        boxesn = nms(boxes, 1, 0.5)
        # print(boxes)
        image = Image.open(path[0]).convert("RGB")
        image = image.resize((416, 416), Image.Resampling.LANCZOS)
        # print(boxesn)
        plot_image(image, boxesn, 'voc')


def test():
    csv_file = "./data/coco_minitrain2017.csv"
    image_dir = "./data/coco_minitrain_25k/images/train2017"
    label_dir = "./data/coco_minitrain_25k/labels/train2017"

    data_dir = "./data/coco_minitrain_25k"
    train_annotation_file = 'train2017.txt'
    ANCHORS = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]  # Note these have been rescaled to be between [0, 1]


    dataset = CocoDataset(data_dir, train_annotation_file, ANCHORS)
    # img, target = dataset.__getitem__(2)

    S = [13, 26, 52]
    scaled_anchors = torch.tensor(ANCHORS) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y, path in loader:
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            # print(anchor.shape)
            # print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]

        boxesn = nms(boxes, 1, 0.5)
        # print(boxes)
        image = Image.open(path[0]).convert("RGB")
        image = image.resize((416, 416), Image.Resampling.LANCZOS)
        plot_image(np.array(image), boxesn, 'coco')



def plot_box():
    image_path = "./data/coco_minitrain_25k/images/train2017/000000484136.jpg"
    label_path = "./data/coco_minitrain_25k/labels/train2017/000000484136.txt"
    
    bboxes = np.loadtxt(fname=label_path, delimiter=" ", ndmin=2).tolist()
    image = Image.open(image_path).convert("RGB")
    image = image.resize((416, 416), Image.Resampling.LANCZOS)
    for box in bboxes:
        box.insert(1, 1)
    size_w, size_h = image.size
    scale_x, scale_y = 416/size_w, 416/size_h
    for box in bboxes:
        box[2] *= scale_x
        box[3] *= scale_y
        box[4] *= scale_x
        box[5] *= scale_y
    plot_image(np.array(image), bboxes)


if __name__ == "__main__":
    # test()
    # plot_box()
    # test_voc()
    test_dog_cat()


