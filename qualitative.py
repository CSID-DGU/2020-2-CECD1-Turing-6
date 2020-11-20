import os,sys,json,random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from datasets.carpart import *
from albumentations import *
import torch.nn.functional as F
from models.pspnet import PSPNet

base_aug = Compose([Resize(512,512,p=1),], p=1)

img_path = "/home/yo0n/workspace2/CarPart/image"
img_target = random.choice([img_path + "/" + i for i in os.listdir(img_path)])
imgor = Image.open(img_target).convert("RGB")
max_size = max(imgor.height, imgor.width)
pad = (max_size - min(imgor.height, imgor.width))//2
imgor = np.uint8(resize_with_padding(imgor, (max_size, max_size)))
img = (torch.tensor(base_aug(image=imgor)['image']).unsqueeze(0).permute(0,3,1,2)/255.).float()

#model load
pspnet = torch.load("./ckpt/pspnet.pth").eval()
deeplabv3x = torch.load("./ckpt/deeplabv3.pth").eval()
models = [pspnet,deeplabv3x]

img = img.cuda()
plt.subplot(len(models)+1,1,1)
plt.imshow(imgor)

for idx,model in enumerate(models):
    model = model.cuda()
    out = model(img)
    out = F.interpolate(out,size=(max_size, max_size),mode="bilinear",align_corners=True)
    pred = np.argmax(out[0].detach().cpu().numpy(), axis=0)

    plt.subplot(len(models)+1,1,idx+2)
    plt.title(model.__class__.__name__)
    plt.imshow(pred)

plt.show()
