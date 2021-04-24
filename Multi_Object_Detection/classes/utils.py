import torch
import torchvision
import torch.nn as nn

from copy import deepcopy
import numpy as np
import matplotlib.pylab as plt

from PIL import Image, ImageDraw, ImageFont

COLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8")
# FNT = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 16)
FNT = ImageFont.truetype('arial.ttf', 16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rescale_bbox(bb, W, H):
    x, y, w, h = bb
    return [x * W, y * H, w * W, h * H]


def show_img_box(img, targets, COCO_Names=None):
    if torch.is_tensor(img):
        # img = torchvision.transforms.ToPILImage(img)
        img = torchvision.transforms.functional.to_pil_image(img)
    if torch.is_tensor(targets):
        targets = targets.numpy()[:, 1:]

    W, H = img.size
    draw = ImageDraw.Draw(img)
    for tg in targets:
        id_ = int(tg[0])
        bbox = tg[1:]
        bbox = rescale_bbox(bbox, W, H)
        xc, yc, w, h = bbox
        color = [int(c) for c in COLORS[id_]]
        name = COCO_Names[id_]
        draw.rectangle(((xc - w / 2, yc - h / 2), (xc + w / 2, yc + h / 2)), outline=tuple(color), width=3)
        draw.text((xc - w / 2, yc - h / 2), name, font=FNT)
    plt.imshow(np.array(img))


# defining the loss function for YOLO-V3, ==> loss = loss_x + loss_y + loss_w + loss_h + loss_obj + loss_cls

def get_loss_batch(outputs, targets, params_loss, opt=None):
    ignore_thres = params_loss["ignore_thres"]
    scaled_anchors = params_loss["scaled_anchors"]
    num_anchors = params_loss["num_anchors"]
    mse_loss = params_loss["mse_loss"]
    bce_loss = params_loss["bce_loss"]
    num_yolos = params_loss["num_yolos"]
    obj_scale = params_loss["obj_scale"]
    noobj_scale = params_loss["noobj_scale"]

    loss = 0.0
    for yolo_ind in range(num_yolos):
        yolo_output = outputs[yolo_ind]
        batch_size, num_bbxs, _ = yolo_output.shape
        gz_2 = num_bbxs / num_anchors
        grid_size = int(np.sqrt(gz_2))

        yolo_output = yolo_output.view(batch_size, num_anchors, grid_size, grid_size, -1)

        # output of the model
        pred_boxes = yolo_output[:, :, :, :, :4]
        x, y, w, h = transform_bbox(pred_boxes, scaled_anchors[yolo_ind])
        pred_conf = yolo_output[:, :, :, :, 4]
        pred_cls_prob = yolo_output[:, :, :, :, 5:]

        # target to compare for loss
        yolo_targets = get_yolo_targets({"pred_cls_prob": pred_cls_prob,
                                         "pred_boxes": pred_boxes,
                                         "targets": targets,
                                         "anchors": scaled_anchors[yolo_ind],
                                         "ignore_thres": ignore_thres})
        obj_mask = yolo_targets["obj_mask"]
        noobj_mask = yolo_targets["noobj_mask"]
        tx = yolo_targets["tx"]
        ty = yolo_targets["ty"]
        tw = yolo_targets["tw"]
        th = yolo_targets["th"]
        tcls = yolo_targets["tcls"]
        t_conf = yolo_targets["t_conf"]

        loss_x = mse_loss(x[obj_mask], tx[obj_mask])
        loss_y = mse_loss(y[obj_mask], ty[obj_mask])
        loss_w = mse_loss(w[obj_mask], tw[obj_mask])
        loss_h = mse_loss(h[obj_mask], th[obj_mask])

        loss_conf_obj = bce_loss(pred_conf[obj_mask], t_conf[obj_mask])
        loss_conf_noobj = bce_loss(pred_conf[noobj_mask], t_conf[noobj_mask])

        loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj

        loss_cls = bce_loss(pred_cls_prob[obj_mask], tcls[obj_mask])

        loss += loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

    if opt:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()


def transform_bbox(bbox, anchors, *args, **kwargs):
    x = bbox[:, :, :, :, 0]
    y = bbox[:, :, :, :, 1]
    w = bbox[:, :, :, :, 2]
    h = bbox[:, :, :, :, 3]

    anchor_w = anchors[:, 0].view((1, 3, 1, 1))
    anchor_h = anchors[:, 1].view((1, 3, 1, 1))

    x = x - x.floor()
    y = y - y.floor()
    w = torch.log(w / anchor_w + 1e-16)
    h = torch.log(h / anchor_h + 1e-16)
    return x, y, w, h


def get_yolo_targets(params, *args, **kwargs):
    pred_cls_prob = params["pred_cls_prob"]
    pred_boxes = params["pred_boxes"]
    targets = params["targets"]
    anchors = params["anchors"]
    ignore_thres = params["ignore_thres"]

    batch_size = pred_boxes.size(0)
    num_anchors = pred_boxes.size(1)
    grid_size = pred_boxes.size(2)
    num_cls = pred_cls_prob.size(-1)

    sizeT = batch_size, num_anchors, grid_size, grid_size
    obj_mask = torch.zeros(sizeT, device=device, dtype=torch.bool)  # torch.uint8 and .bool()
    noobj_mask = torch.ones(sizeT, device=device, dtype=torch.bool)

    tx = torch.zeros(sizeT, device=device, dtype=torch.float32)
    ty = torch.zeros(sizeT, device=device, dtype=torch.float32)
    tw = torch.zeros(sizeT, device=device, dtype=torch.float32)
    th = torch.zeros(sizeT, device=device, dtype=torch.float32)

    # update sizeT
    sizeT = batch_size, num_anchors, grid_size, grid_size, num_cls
    tcls = torch.zeros(sizeT, device=device, dtype=torch.float32)

    target_bboxes = targets[:, 2:] * grid_size
    t_xy = target_bboxes[:, :2]
    t_wh = target_bboxes[:, 2:]
    t_x, t_y = t_xy.t()
    t_w, t_h = t_wh.t()

    grid_i, grid_j = t_xy.long().t()

    iou_with_anchors = [get_iou_WH(anchor, t_wh) for anchor in anchors]
    iou_with_anchors = torch.stack(iou_with_anchors)
    best_iou_wa, best_anchor_ind = iou_with_anchors.max(0)

    batch_inds, target_labels = targets[:, :2].long().t()
    obj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 1
    noobj_mask[batch_inds, best_anchor_ind, grid_j, grid_i] = 0

    for ind, iou_wa in enumerate(iou_with_anchors.t()):
        noobj_mask[batch_inds[ind], iou_wa > ignore_thres, grid_j[ind], grid_i[ind]] = 0

    tx[batch_inds, best_anchor_ind, grid_j, grid_i] = t_x - t_x.floor()
    ty[batch_inds, best_anchor_ind, grid_j, grid_i] = t_y - t_y.floor()

    anchor_w = anchors[best_anchor_ind][:, 0]
    anchor_h = anchors[best_anchor_ind][:, 1]

    tw[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_w / anchor_w + 1e-16)
    th[batch_inds, best_anchor_ind, grid_j, grid_i] = torch.log(t_h / anchor_h + 1e-16)

    tcls[batch_inds, best_anchor_ind, grid_j, grid_i, target_labels] = 1

    output = dict(obj_mask=obj_mask,
                  noobj_mask=noobj_mask,
                  tx=tx,
                  ty=ty,
                  tw=tw,
                  th=th,
                  tcls=tcls,
                  t_conf=obj_mask.float())
    return output


def get_iou_WH(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + (w2 * h2 - inter_area)
    return inter_area / union_area


# function for training the model

def loss_epoch(model, params_loss, dataset_dl, sanity_check=False, opt=None):
    epoch_loss = 0.0
    epoch_metric = {}
    len_data = len(dataset_dl.dataset)

    for xb, yb, _ in dataset_dl:
        yb = yb.to(device)
        _, output = model(xb.to(device))
        loss_b = get_loss_batch(outputs=output, targets=yb, params_loss=params_loss, opt=opt)
        epoch_loss += loss_b
        if sanity_check:
            break
        del xb, yb
        torch.cuda.empty_cache()
    loss = epoch_loss / float(len_data)
    return loss


def get_lr(opt):
    for param in opt.param_groups:
        return param["lr"]


def train_val(model, params):
    num_epochs = params["num_epochs"]
    params_loss = params["params_loss"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    sanity_check = params["sanity_check"]
    lr_scheduler = params["lr_scheduler"]
    path2weight = params["path2weights"]

    loss_history = dict(train=[], val=[])

    best_model_weights = deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print(f"Epoch: {epoch + 1}/{num_epochs}, current learning rate: {current_lr}")

        model.train()
        train_loss = loss_epoch(model=model, params_loss=params_loss, dataset_dl=train_dl, sanity_check=sanity_check,
                                opt=opt)
        loss_history["train"].append(train_loss)

        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model=model, params_loss=params_loss, dataset_dl=val_dl, sanity_check=sanity_check,
                                  opt=None)
            loss_history["val"].append(val_loss)

        print(f"train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = deepcopy(model.state_dict())
            torch.save(model.state_dict(), path2weight)
            print("==> copied best model weights!")

        lr_scheduler.step(val_loss)

        if current_lr != get_lr(opt):
            print("==> Loading the best model weights...!!!")
            model.load_state_dict(best_model_weights)

        print("-" * 50)
    model.load_state_dict(best_model_weights)
    return model, loss_history


def plotResults(title, num_epochs, df, xLabel, yLabel, train_Label, val_Label):
    plt.title(title)
    plt.plot(range(1, num_epochs + 1), df["train"], label=train_Label)
    plt.plot(range(1, num_epochs + 1), df["val"], label=val_Label)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.legend()
    plt.show()


# For Deploying the model

def NonMaxSuppression(bbox_pred, obj_thres=0.5, nms_thres=0.5):
    bbox_pred[..., :4] = xywh2xyxy(bbox_pred[..., :4])
    output = [None] * len(bbox_pred)
    for ind, bb_pr in enumerate(bbox_pred):
        bb_pr = bb_pr[bb_pr[:, 4] >= obj_thres]
        if not bb_pr.size(0):
            continue

    score = bb_pr[..., 4] * bb_pr[:, 5:].max(1)[0]
    bb_pr = bb_pr[(-score).argsort()]
    cls_probs, cls_preds = bb_pr[:, 5:].max(1, keepdim=True)
    detections = torch.cat((bb_pr[:, :5], cls_probs.float(), cls_preds.float()), 1)

    bbox_nms = []
    while detections.size(0):
        high_iou_inds = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres

        cls_match_inds = detections[0, -1] == detections[:, -1]
        supp_inds = high_iou_inds & cls_match_inds

        ww = detections[supp_inds, 4]
        detections[0, :4] = (ww * detections[supp_inds, :4]).sum(0) / ww.sum()
        bbox_nms += [detections[0]]
        detections = detections[~supp_inds]
        if bbox_nms:
            output[ind] = torch.stack(bbox_nms)
            output[ind] = xywh2xyxy(output[ind])
        return output


def xywh2xyxy(xywh):
    xyxy = xywh.new(xywh.shape)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2.0
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2.0
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2.0
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2.0
    return xyxy


def xyxyh2xywh(xyxy, img_size=416):
    xywh = torch.zeros(xyxy.shape[0], 6)
    xywh[:, 2] = (xyxy[:, 0] + xyxy[:, 2]) / 2. / img_size
    xywh[:, 3] = (xyxy[:, 1] + xyxy[:, 3]) / 2. / img_size
    xywh[:, 5] = (xyxy[:, 2] - xyxy[:, 0]) / img_size
    xywh[:, 4] = (xyxy[:, 3] - xyxy[:, 1]) / img_size
    xywh[:, 1] = xyxy[:, 6]
    return xywh


def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) \
                 * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1.0) * (b1_y2 - b1_y1 + 1.0)
    b2_area = (b2_x2 - b2_x1 + 1.0) * (b2_y2 - b2_y1 + 1.0)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou


def bbox_iou():
    pass
