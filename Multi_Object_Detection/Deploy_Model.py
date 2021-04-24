import torch
import os
from Multi_Object_Detection.classes.utils import *
from Multi_Object_Detection.classes.Model import DarkNet
from Multi_Object_Detection.classes.Dataset import dataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "../Data/coco"
model = DarkNet(os.path.join(path, "yolov3.cfg")).to(device)

train_dl, val_dl, coco_train, coco_val = dataLoader(path)

img, tag, _ = coco_val[4]
print(img.shape)
print(tag.shape)

show_img_box(img, tag)

path2weights ="./model/yolov3-tiny.weights"
model.load_state_dict(torch.load(path2weights))

model.eval()
print(model)

with torch.no_grad():
    out, _ = model(img.unsqueeze(0).to(device))

print(out.shape)

img_size = 416
out_nms = NonMaxSuppression(out.cpu())
print(out_nms[0].shape)
show_img_box(img, out_nms[0])