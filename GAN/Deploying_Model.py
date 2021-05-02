import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from GAN.classes.Model import *
from GAN.classes.Utils import *
from GAN.classes.Dataset import *

params_gen = dict(nz=100, ngf=64, noc=3)
model_gen = Generator(params_gen).to(device)

model_gen.load_state_dict(torch.load("./model/weights_gen.pt"))
model_gen.eval()

with torch.no_grad():
    fixed_noise = torch.randn(16, 100, 1, 1, device=device)
    img_fake = model_gen(fixed_noise).detach().cpu()

print(img_fake.shape)

img = make_grid(img_fake, nrow=4, padding=2)

plt.figure(figsize=(10, 10))
plt.imshow(img.permute(1,2,0))
plt.axis("off")
plt.show()
