import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from albumentations import *

from datasets.carpart import *

model_path = "pkcDjango/services/25.pth"
prefix = "media/tempFiles/"
base_aug = Compose([Resize(512, 512, p=1), ], p=1)


def img_seg(file):
    img_path = file.path
    save_path = prefix + os.path.splitext(file.fileName())[0] + "_seg.png"

    imgor = Image.open(img_path).convert("RGB")
    max_size = max(imgor.height, imgor.width)
    pad = (max_size - min(imgor.height, imgor.width)) // 2
    imgor = np.uint8(resize_with_padding(imgor, (max_size, max_size)))
    img = (torch.tensor(base_aug(image=imgor)['image']).unsqueeze(0).permute(0, 3, 1, 2) / 255.).float()

    model = torch.load(model_path).eval()

    img = img.cuda()
    model = model.cuda()
    out = model(img)
    out = F.interpolate(out, size=(max_size, max_size), mode="bilinear", align_corners=True)
    pred = np.argmax(out[0].detach().cpu().numpy(), axis=0)

    return [pred, save_path]
# img_seg()
