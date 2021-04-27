import torch
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
    for params in vgg19.parameters():
        params.requires_grad_(False)
    return vgg19


if __name__ == '__main__':
    model = load_model()
    print(model)
    for name in model.children():
        print(name)