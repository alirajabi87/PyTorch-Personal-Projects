import matplotlib.pyplot as plt
import torch.optim

from NeuralStyleTransfer.classes.Dataset import *
from NeuralStyleTransfer.classes.Utils import *
from NeuralStyleTransfer.classes.Model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    path2content = "../Data/StyleTransfer/content_N.jpg"
    path2style = "../Data/StyleTransfer/style_N.jpg"
    content_img, style_img = loading_images(path2content, path2style)
    w_i, h_i = content_img.size
    content_tensor, style_tensor = transformer(content_img, style_img)
    vgg19 = load_model()

    feature_layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
                      '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
    con_tensor = content_tensor.unsqueeze(0).to(device)
    sty_tensor = style_tensor.unsqueeze(0).to(device)

    content_features = get_features(con_tensor, vgg19, feature_layers)
    style_features = get_features(sty_tensor, vgg19, feature_layers)

    input_tensor = con_tensor.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([input_tensor], lr=1e-2)
    style_layers_dict = {'conv1_1': 0.75,
                         'conv2_1': 0.5,
                         'conv3_1': 0.25,
                         'conv4_1': 0.25,
                         'conv5_1': 0.25}
    params_train = dict(num_epochs=701,
                        optimizer=optimizer,
                        content_weight=1e1,
                        style_weight=1e5,
                        content_layer="conv5_1",
                        style_layers=style_layers_dict,
                        model=vgg19,
                        feature_layers=feature_layers,
                        input_tensor=input_tensor,
                        content_features=content_features,
                        style_features=style_features)
    img = train(params_train)
    img = transforms.Resize((h_i, w_i))(img)
    img = imgtensor2pil(img)
    img.save("../Data/StyleTransfer/"+"Alineg.jpg", "JPEG")
    plt.figure()
    plt.imshow(img)

    plt.savefig("../Data/StyleTransfer/Alineg.png", dpi=300, edgecolor='w')
    plt.show()


if __name__ == '__main__':
    main()
