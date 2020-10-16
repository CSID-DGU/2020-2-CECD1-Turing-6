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

#basic setting
num_epochs = 30
print_iter = 25
batch_size = 2
vis_result= True
validation_ratio = 0.05
startlr = 1e-4

#model load
model = model = PSPNet(n_classes=30, n_blocks=[3, 4, 6, 3], pyramids=[6, 3, 2, 1]).cuda()
#model = torch.load("./ckpt/25.pth")
model_name = model.__class__.__name__

#dataset load
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


train_dataset = CarPart(transform=aug)
dataiter,dataset_len = len(train_dataset)//batch_size,len(train_dataset)
train_len = int(dataset_len*(1-validation_ratio))
train_dataset, validate_dataset = torch.utils.data.random_split(train_dataset, [train_len, dataset_len-train_len])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=1,shuffle=True, num_workers=0)
print("trainset lenght :: ",len(train_dataset))

#loss
criterion  = nn.CrossEntropyLoss()
seg_criterion = nn.NLLLoss2d(weight=None)
cls_criterion = nn.BCEWithLogitsLoss(weight=None)

#optim setting
optimizer = optim.RMSprop(model.parameters(), lr=startlr, weight_decay=1e-5, momentum=0)
scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=startlr, max_lr=startlr*3, step_size_up=2000, mode='triangular2' , gamma=0.9994,cycle_momentum=False )
opt = SWA(optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)

global_iter = 0
for epoch in range(num_epochs):
    losses = list()
    print("optim lr : ",optimizer.param_groups[0]['lr'])
    for iteration,sample in enumerate(train_loader):
        global_iter+=1

        img, label = sample #torch.Size([4, 368, 640, 3]) torch.Size([4, 368, 640]) torch.Size([4, 368, 640])
        iter_batch_size= img.size(0)
        if(iter_batch_size==1):
            continue
        img = img.permute(0,3,1,2).float() / 255.
        label = label.view(-1,512,512).long()
        img,label = img.cuda(),label.cuda()

        size_in = img.size()
        out = model(img)
        aux = F.interpolate(out[0],size=(size_in[2], size_in[3]),mode="bilinear",align_corners=True)
        h = F.interpolate(out[1],size=(size_in[2], size_in[3]),mode="bilinear",align_corners=True)

        loss_, inter_loss = criterion(h, label), criterion(aux, label)
        loss = loss_ + 0.4 * inter_loss
        #print(aux.shape, h.shape, label.shape)

        #loss     = criterion(h, label )

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("  avg loss :: ",sum(losses)/len(losses), end = "\r")

        if(iteration%print_iter == 0):

            pred = np.argmax(h[0].detach().cpu().numpy(), axis=0)
            #print(h[0].shape, pred.shape , pred.min(), pred.max())
            img_0 = img[0].permute(1,2,0).squeeze().cpu().numpy()

            plt.subplot(2,1,1)
            plt.imshow(img_0)
            plt.subplot(2,1,2)
            plt.imshow(pred)
            plt.show(block=False)
            plt.pause(3)
            plt.close()

    print(str(epoch)," :: ",str(iteration), "/",dataiter,"\n  loss     :: ",loss.item())

    scheduler.step()
    if(epoch %5 ==0):
        torch.save(model,"./ckpt/"+str(epoch)+".pth")
