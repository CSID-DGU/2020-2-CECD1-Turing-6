from __future__ import print_function

import os
import os.path as osp
import sys
import json
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import matplotlib.pyplot as plt

class CarPart(data.Dataset):
    def __init__(
        self,
        root="/sdb/CarPart",
        transform=None,
        target_transform=None,
        dataset_name="Carpart",
    ):
        self.root = root
        self.image_root = self.root+"/image"
        self.label_root = self.root+"/label"
        self.transform = transform
        self.mean_rgb = np.array([123.68, 116.779, 103.939])
        self.classMap = {'Emblem': 0, 'Quarter Glass (R)': 1, 'Rear Door (R)': 2, 'Rear Wheel (R)': 3, 'Rear bumper': 4, 'Front Door (R)': 5, '[R] A Pillar (Front Pillar)': 6, 'Trunk Door': 7, 'Rear Door Handle/Catch (R)': 8, 'Front Door Glass (R)': 9, 'Tail Lamp (L)': 10, '[R] Roof Rail': 11, 'Front Fender (R)': 12, 'Tail Lamp (R)': 13, '[R] D pillar (Quarter Panel)': 14, 'Rear Door Glass (R)': 15, 'Rear Glass': 16, '[R] B pillar (Center Panel)': 17, 'Rear Fender (R)': 18}


        self.images = [self.image_root + "/" + i for i in os.listdir(self.image_root)]
        self.partJson = [self.label_root + "/" + i for i in os.listdir(self.label_root)]
        self.jsons = list()
        for i in self.partJson:
            with open(i, 'r') as f:
                data = json.load(f)
            self.jsons.append(data)

    def parseLabel(self, id):
        points = list()
        cls = list()
        for json in self.jsons:
            try:
                raw=json[id]
                for shape in raw['regions']:
                    cls_ = self.classMap[shape['region_attributes']['part']]+1
                    pts_x = shape['shape_attributes']['all_points_x']
                    pts_y = shape['shape_attributes']['all_points_y']
                    pts = np.stack([pts_x, pts_y],axis=1)
                    points.append(pts)
                    cls.append(cls_)
            except:
                continue
        return cls, points

    def __getitem__(self, index):
        img_path = self.images[index]
        img_id = img_path.split("/")[-1]

        cls, points = self.parseLabel(img_id)
        img = Image.open(img_path).convert("RGB")
        mask = np.zeros((img.height, img.width))

        for i,pts in enumerate(points):
            cv2.fillPoly(mask, np.int32([pts]) , cls[i])

        plt.subplot(2,1,1)
        plt.imshow(img)
        plt.subplot(2,1,2)
        plt.imshow(mask)
        plt.show()

        if self.transform is not None:
            img = self.transform(img)

        img = (np.array(img) - self.mean_rgb).transpose(2, 0, 1)
        img = torch.from_numpy(img.astype(np.float32))

        return img, target

    def __len__(self):
        return len(self.images)

if __name__=="__main__":
    import random

    dataset = CarPart()
    img, target = dataset[2000]
