import os
import numpy as np
import torch
import sys
from PIL import Image, ImageOps
import torch.nn.functional as F
from torchvision.utils import save_image
from albumentations import *
from .carpart import *
from pathlib import Path
from .models.pspnet import PSPNet

#os.environ['KMP_DUPLICATE_LIB_OK']='True'

#from pkcDjango.models import File
BASE_DIR = Path(__file__).parent.parent.parent
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
PKG_ROOT = Path(__file__).resolve().parent
#model path .pht
#image_path
img_path = "pkcDjango/services/1-1.PNG"
save_path = "pkcDjango/services/save.PNG"

#input ../img.jpg save ../img_seg.png
def img_seg(file):
    #model_path = str(os.path.join(PKG_ROOT,"5.pth"))
    img_path = str(os.path.join(MEDIA_ROOT,str(file.path)))
    save_path = os.path.splitext(img_path)[0] + "_seg.png"
    #open images
    base_aug = Compose([Resize(512, 512, p=1), ], p=1)
    imgor = Image.open(img_path).convert("RGB")
    max_size = max(imgor.height, imgor.width)
    pad = (max_size - min(imgor.height, imgor.width))//2
    imgor = np.uint8(resize_with_padding(imgor, (max_size, max_size)))
    img = (torch.tensor(base_aug(image=imgor)['image']).unsqueeze(0).permute(0, 3, 1, 2)/255.).float()

    #model load
    #model = torch.load(model_path).eval()
    model = PSPNet(n_classes=30, n_blocks=[3, 4, 6, 3], pyramids=[6, 3, 2, 1])
    model = model.cuda()
    modelp = str(os.path.join(PKG_ROOT,"10_.pth"))
    model.load_state_dict(torch.load(modelp))
    model.eval()

    img = img.cuda()
    model = model.cuda()
    out = model(img)
    out = F.interpolate(out, size=(max_size, max_size), mode="bilinear", align_corners=True)
    pred = np.argmax(out[0].detach().cpu().numpy(), axis=0)

    plt.imshow(pred)
    plt.savefig(save_path)
    