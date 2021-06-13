import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchcontrib.optim import SWA

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

from datasets.carpart import CarPart
from albumentations import *
from models.pspnet import PSPNet
from models.deeplab import DeepLab
from utils.metric import *

#basic setting
batch_size = 1
vis_result= True
num_classes = 24

#model load
#model = model = PSPNet(n_classes=30, n_blocks=[3, 4, 6, 3], pyramids=[6, 3, 2, 1]).cuda()
model = torch.load("./ckpt/25.pth").cuda().eval()
model_name = model.__class__.__name__

#dataset load
aug = Compose([
             Resize(512,512,p=1),
            ], p=1)


test_dataset = CarPart(transform=aug, phase='test')
dataiter,dataset_len = len(test_dataset)//batch_size,len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)

print("testset lenght :: ",len(test_dataset))

label_trues = list()
label_preds = list()

for iteration,sample in enumerate(test_loader):

    img, label = sample #torch.Size([4, 368, 640, 3]) torch.Size([4, 368, 640]) torch.Size([4, 368, 640])
    iter_batch_size= img.size(0)

    img = img.permute(0,3,1,2).float() / 255.
    label = label.view(-1,512,512).long()
    img,label = img.cuda(),label.cuda()

    boundary, pred =  model(img)
    #print(pred.shape, boundary.shape)
    out = pred.squeeze().detach()

    label_trues.append(label.squeeze().cpu().numpy())
    label_preds.append(torch.max(out, axis=0)[1].cpu().numpy())

score = scores(label_trues, label_preds, num_classes)

print("Mean IoU     : " ,score[0]["Mean IoU"])
print("Overall Acc  : " ,score[0]["Overall Acc"])
