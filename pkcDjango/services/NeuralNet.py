import os
import sys
import json
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from pytorch.datasets.carpart import *
from albumentations import *
import torch.nn.functional as F
# from pytorch.models.pspnet import PSPNet
from torchvision.utils import save_image
import torchvision
from pkcDjango.models import File

# model path .pht
model_path = "C:\\Users\\Administrator\\PycharmProjects\\pkcDjango\\pkcDjango\\services\\ckpt"
# image_path
img_path = "/home/yo0n/workspace2/CarPart/image"
save_paht = "saved.png"
base_aug = Compose([Resize(512, 512, p=1), ], p=1)


# input ../img.jpg save ../img_seg.png
def img_seg(file):
    img_path = file.fileName()
    save_path = os.path.splitext(img_path)[0] + "_seg.png"
    # open images
    imgor = Image.open(img_path).convert("RGB")
    max_size = max(imgor.height, imgor.width)
    pad = (max_size - min(imgor.height, imgor.width)) // 2
    imgor = np.uint8(resize_with_padding(imgor, (max_size, max_size)))
    img = (torch.tensor(base_aug(image=imgor)['image']).unsqueeze(
        0).permute(0, 3, 1, 2) / 255.).float()

    # model load
    model = torch.load(model_path).eval()

    img = img.cuda()
    model = model.cuda()
    out = model(img)
    out = F.interpolate(out, size=(max_size, max_size),
                        mode="bilinear", align_corners=True)
    pred = np.argmax(out[0].detach().cpu().numpy(), axis=0)

    save_image(pred, save_path)
