from __future__ import print_function

import os
import os.path as osp
import sys
import json
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageOps
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

def resize_with_padding(img, expected_size):
    img.thumbnail((expected_size[0], expected_size[1]))
    # print(img.size)
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)

class CarPart(data.Dataset):
    def __init__(
        self,
        root="D:\\Documents\\carpart\\CarPart",
        transform=None,
        target_transform=None,
        dataset_name="Carpart",
    ):
        self.root = root
        self.image_root = self.root+"/image"
        self.label_root = self.root+"/label2"
        self.transform = transform
        self.mean_rgb = np.array([123.68, 116.779, 103.939])
        self.classMap = {
        'Quarter Glass (R)': 1,'Quarter Glass (L)':1,
        'Rear Door (R)': 2,'Rear Door (L)':2,
        'Rear Wheel (R)': 3,'Rear Wheel (L)':3,
        'Rear bumper': 4,
        'Front Door (R)': 5,'Front Door (L)': 5,
        '[R] A Pillar (Front Pillar)': 6, '[L] A Pillar (Front Pillar)':6,
        'Trunk Door': 7,
         'Rear Door Handle/Catch (R)': 8,'Front Door Handle/Catch (L)':8,
         'Front Door Glass (R)': 9,'Front Door Glass (L)':9,
         'Tail Lamp (L)': 10, 'Tail Lamp (R)': 10,
         '[R] Roof Rail': 11,'[L] Roof Rail':11,
         'Front Fender (R)': 12,'Front Fender (L)':12,
         '[R] D pillar (Quarter Panel)': 13,'[L] D pillar (Quarter Panel)':13,
         'Rear Door Glass (R)': 14,'Rear Door Glass (L)': 14,
          'Rear Light':15,
          '[R] B pillar (Center Panel)': 16,'[L] B pillar (Center Panel)':16,
          'Rear Fender (R)': 17,'Rear Fender (L)':17,
          'Front bumper':18,
          'Rediator Grille':19,
          'Head Lamp (R)':20,'Head Lamp (L)':20,
          'Bonnet/Hood':21,
          '[L] C pillar (Rear Pillar)':22, '[R] C pillar (Rear Pillar)':22,
          'Quarter Door Glass (R)' : 23, 'Quarter Door Glass (L)' : 23,
          }

        self.num_classes = max(self.classMap.values())
        self.images = [self.image_root + "/" + i for i in os.listdir(self.image_root)]
        self.partJson = [self.label_root + "/" + i for i in os.listdir(self.label_root)]
        self.jsons = list()
        for i in self.partJson:
            with open(i, 'r') as f:
                data = json.load(f)
            self.jsons.append(data)

        cls_dist = np.zeros(31)
        cls_cnt = list()
        parts_in_image =list()
        valid_images = list()
        for i in range(len(self.images)):
            img_path = self.images[i]
            img_id = img_path.split("/")[-1]
            cls, label =  self.parseLabel(img_id, 0, 0)
            if len(cls)>0:
                parts_in_image.append(len(cls))
                cls_dist[cls] += 1
                cls_cnt += cls
                valid_images.append( self.images[i] )
        #plt.hist(cls_cnt,bins=30)
        #plt.show()
        self.valid_images = valid_images

    def parseLabel(self, id, max_size, pad):
        points = list()
        cls = list()
        parts = list()
        missed = list()
        retouch = list()
        mask = np.zeros((max_size, max_size))
        for json in self.jsons:
            try:
                raw=json[id]
                for shape in raw['regions']:
                    part = shape['region_attributes']['part']
                    #print( " -> " ,part)
                    parts.append(part)
                    try:
                        cls_ = self.classMap[shape['region_attributes']['part']]#+1
                    except:
                        missed.append(shape['region_attributes']['part'])
                    shape_name = shape['shape_attributes']['name']
                    if 'ellipse' in shape_name:
                        continue
                    else:
                        pts_x = shape['shape_attributes']['all_points_x']
                        pts_y = shape['shape_attributes']['all_points_y']
                        pts = np.stack([pts_x, pts_y],axis=1)
                        pts = np.int32([pts])
                        pts[:,:,1] = pts[:,:,1] + pad
                        cv2.fillPoly(mask,  pts , cls_)
                        points.append(pts)
                        #if part == "Emblem":
                        #    retouch.append([pts, cls_])
                    cls.append(cls_)
            except:
                continue

        if len(retouch) > 0:
            for re in retouch:
                try:
                    cv2.fillPoly(mask,  re[0] , re[1])
                except:
                    print('retouch error')

        #print(set(parts))

        return cls, mask

    def __getitem__(self, index):

        img_path = self.valid_images[index]
        img_id = img_path.split("/")[-1]

        img = Image.open(img_path).convert("RGB")
        max_size = max(img.height, img.width)
        pad = (max_size - min(img.height, img.width))//2
        #print("========")
        cls, mask = self.parseLabel(img_id, max_size, pad)
        img = np.uint8(resize_with_padding(img, (max_size, max_size)))


        if self.transform is not None:
            transformed = self.transform(image = img, mask=mask)
            img = transformed['image']

            mask = transformed['mask']

        return img, mask

    def __len__(self):
        return len(self.valid_images)

if __name__=="__main__":
    import random
    from albumentations import *


    aug = Compose([
                 HorizontalFlip(0.5),
                 OneOf([
                    MotionBlur(p=0.2),
                    MedianBlur(blur_limit=3, p=0.1),
                    ISONoise(p=0.3),
                    Blur(blur_limit=3, p=0.1),], p=0.5),
                 RandomBrightnessContrast(brightness_limit=(-0.3,0.4),p=0.5),
                 ShiftScaleRotate(shift_limit=0.125, scale_limit=(0.25,0.5), rotate_limit=2,border_mode=cv2.BORDER_CONSTANT, p=0.5),
                 OneOf([HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
                        ToSepia(p=0.05),], p=0.3),
                 OneOf([RandomShadow(p=0.2),
                        RandomRain(blur_value=2,p=0.4),
                        RandomFog(fog_coef_lower=0.1,fog_coef_upper=0.2, alpha_coef=0.25, p=0.2),], p=0.25),
                 Resize(512,512,p=1),
                 Cutout(num_holes=5, max_h_size=30, max_w_size=30,p=0.5)
                ], p=1)


    dataset = CarPart(transform=aug)
    print(len(dataset))
    img, mask = dataset[400]

    print(np.unique(mask))

    if True:
        plt.subplot(2,2,1)
        plt.imshow(img)
        plt.subplot(2,2,2)
        plt.imshow(mask)
        plt.show()
        #cv2.imshow('mask', mask)
        #cv2.waitKey(0)
