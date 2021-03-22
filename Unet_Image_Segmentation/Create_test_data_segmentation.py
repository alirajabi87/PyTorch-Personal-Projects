import os
import random
import numpy as np
from glob import glob
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
import cv2 as cv

path = "../Data/Image_segmentation/"

images = glob(os.path.join(path + "train/*.jp*g"))
masks = glob(os.path.join(path+"train_masks/*.gif"))

print(len(images))
# print(len(masks))
# for i in range(int(0.1*len(images))):
#     images = glob(os.path.join(path + "train/*.jp*g"))
#     masks = glob(os.path.join(path + "train_masks/*.gif"))
#
#     img_name = random.choice(images)
#
#     path_mask = img_name.split("\\")[0]+"_masks/"
#     name = img_name.split("\\")[-1].split(".")[0]
#
#     mask_name = os.path.join(path_mask + name +"_mask.gif")
#
#     with Image.open(img_name) as img:
#         img.save(os.path.join(path + "test/" + name+".jpg"))
#
#     with Image.open(mask_name) as mask:
#         mask.save(os.path.join(path + "test_masks/" + name+"_mask.gif"))
#
#     os.remove(mask_name)
#     os.remove(img_name)


# img = image.imread(os.path.join(path_mask+ name +"_mask.gif"))
# plt.imshow(img)
# plt.show()
#
# img = image.imread(img_name)
# plt.imshow(img)
# plt.show()