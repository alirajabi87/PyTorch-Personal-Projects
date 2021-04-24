from Multi_Object_Detection.classes.Model_Utils import *
import os
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DarkNet(nn.Module):
    def __init__(self, config_path, img_dim=416):
        super(DarkNet, self).__init__()
        self.block_list = parse_model_config(config_path)
        self.hyperparams, self.module_list = create_layers(self.block_list)
        self.img_size = img_dim

    def forward(self, x):
        img_dim = x.shape[2]
        layer_outputs, yolo_outputs = [], []
        for block, module in zip(self.block_list[1:], self.module_list):
            if block["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif block["type"] == "shortcut":
                layer_ind = int(block["from"])
                x = layer_outputs[-1] + layer_outputs[layer_ind]
            elif block["type"] == "yolo":
                x = module[0](x)
                yolo_outputs.append(x)
            elif block["type"] == "route":
                x = torch.cat([layer_outputs[int(l_i)] for l_i in block["layers"].split(",")], 1)
            layer_outputs.append(x)
        yolo_out_cat = torch.cat(yolo_outputs, 1)
        return yolo_out_cat, yolo_outputs


if __name__ == '__main__':
    path = "../../Data/coco"
    # blocks_list = parse_model_config(os.path.join(path, "yolov3.cfg"))
    # # print(blocks_list[0])
    # hy_pa, m_l = create_layers(blocks_list)
    # print(hy_pa)
    # print(m_l)
    model = DarkNet(os.path.join(path, "yolov3.cfg")).to(device)
    # print(model)

    dummy_img = torch.rand(1,3, 416, 416).to(device)
    with torch.no_grad():
        dummy_out_cat, dummy_output = model(dummy_img)
        print(dummy_out_cat.shape)
        print(dummy_output[0].shape, dummy_output[1].shape, dummy_output[2].shape)