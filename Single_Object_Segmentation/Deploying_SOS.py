import matplotlib.pyplot as plt
import torch
import numpy as np

from torchvision.transforms.functional import to_tensor
from PIL import Image
import os

from Single_Object_Segmentation.classes.Model import SegNet
from Single_Object_Segmentation.classes.Dataset import show_img_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    h, w = 128, 192
    BASE_DIR = '../Data/FETAL/test_set/'
    imgList = [pp for pp in os.listdir(BASE_DIR)][:-1]
    print(len(imgList))

    rndImg = np.random.choice(imgList, 4)

    params_model = dict(input_shape=(1, h, w),
                        initial_filters=16,
                        num_outputs=1)
    model = SegNet(params_model).to(device)

    model.load_state_dict(torch.load('./model/weights.pt'))
    model.eval()

    for fn in rndImg:
        path2Img = os.path.join(BASE_DIR, fn)
        img = Image.open(path2Img)

        img = img.resize((w, h))
        img_t = to_tensor(img).unsqueeze(0).to(device)
        pred = torch.sigmoid(model(img_t))[0]
        mask_pred = (pred[0] >= 0.5).cpu()

        img = np.array(img)

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(img, cmap='gray')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_pred, cmap='gray')

        plt.subplot(1, 3, 3)
        show_img_mask(img, mask_pred)

    plt.show()

if __name__ == '__main__':
    main()