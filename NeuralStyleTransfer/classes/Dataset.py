import matplotlib.pyplot as plt
import torch
from PIL import Image

import torchvision.transforms as transforms
from NeuralStyleTransfer.classes.Utils import *

def loading_images(path2content, path2style):
    content_img = Image.open(path2content)
    style_img = Image.open(path2style)

    return content_img, style_img


def show_content_style(content_img, style_img):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(content_img)
    plt.subplot(1, 2, 2)
    plt.imshow(style_img)
    plt.show()


def transformer(con_img, sty_img):
    h, w = 256, 384
    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_rgb, std=std_rgb)
    ])
    return transform(con_img), transform(sty_img)



if __name__ == '__main__':
    path2content = "../../Data/StyleTransfer/content.jpg"
    path2style = "../../Data/StyleTransfer/style.jpg"
    content_img, style_img = loading_images(path2content, path2style)
    content_img, style_img = transformer(content_img, style_img)
    show_content_style(imgtensor2pil(content_img), imgtensor2pil(style_img))
