import torch

from Video_Classification.Classes.Utils import *
from Video_Classification.Classes.Model import get_model
from Video_Classification.Classes.MyDataset import dataLoader

model = get_model(7, "rnn")
model.to(device)
model.eval()

path_weights = "./model/weights.pt"

model.load_state_dict(torch.load(path_weights))

path_vids = "../Data/Videos/hmdb51_org/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0.avi"

frames, vid_len = get_frames(path_vids, n_frame=16)

frames_tr = []
for frame in frames:
    frame = Image.fromarray(frame)
    frames_tr.append(frame)

imgs_tensor = transform_frames(frames_tr, model_type='rnn', mode="validation")
imgs_tensor = imgs_tensor.unsqueeze(0)
print(imgs_tensor.shape, torch.min(imgs_tensor), torch.max(imgs_tensor))

with torch.no_grad():
    out = model(imgs_tensor.to(device)).cpu()
    print(out.shape)
    pred = torch.argmax(out).item()
    print(pred)