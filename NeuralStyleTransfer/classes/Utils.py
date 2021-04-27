import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image


def get_features(x, model, layers):
    features = {}
    for name, layer in enumerate(model.children()):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features


def gram_matrix(x):
    n, c, h, w = x.size()

    x = x.view(n * c, h * w)
    gram = torch.mm(x, x.t())
    return gram


def get_content_loss(pred_features, target_features, layer):
    target = target_features[layer]
    pred = pred_features[layer]
    loss = F.mse_loss(pred, target)
    return loss


def get_style_loss(pred_features, target_features, style_layer_dict):
    loss = 0.0

    for layer in style_layer_dict:
        pred_feat = pred_features[layer]
        pred_gram = gram_matrix(pred_feat)
        n, c, w, h = pred_feat.size()
        target_gram = gram_matrix(target_features[layer])
        layer_loss = style_layer_dict[layer] * F.mse_loss(pred_gram, target_gram)
        loss += layer_loss / (n * c * h * w)

    return loss


def train(params):
    num_epochs = params["num_epochs"]
    content_weight = params["content_weight"]
    style_weight = params["style_weight"]
    content_layer = params["content_layer"]
    style_layers_dict = params["style_layers"]
    opt = params["optimizer"]
    vgg19 = params["model"]
    feature_layers = params["feature_layers"]
    input_tensor = params["input_tensor"]
    content_features = params["content_features"]
    style_features = params["style_features"]

    for epoch in range(num_epochs):
        opt.zero_grad()
        input_features = get_features(input_tensor, model=vgg19, layers=feature_layers)

        content_loss = get_content_loss(input_features, content_features, content_layer)
        style_loss = get_style_loss(input_features, style_features, style_layers_dict)
        neural_loss = content_weight * content_loss + style_weight * style_loss

        neural_loss.backward()
        opt.step()

        if not epoch % 100:
            print(f"epoch: {epoch}, content loss: {content_loss:.2f}, style loss: {style_loss:.2f}")

    return input_tensor[0].cpu()

def imgtensor2pil(img):
    h, w = 256, 384
    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)
    img_c = img.clone().detach()
    img_c *= torch.Tensor(std_rgb).view(3, 1, 1)
    img_c += torch.Tensor(mean_rgb).view(3, 1, 1)
    img_c = img_c.clamp(0, 1)
    img_toPil = to_pil_image(img_c)
    return img_toPil