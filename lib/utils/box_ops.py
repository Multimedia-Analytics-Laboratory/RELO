import torch
from torchvision.ops.boxes import box_area
from torch.nn.functional import l1_loss
import numpy as np


# def box_cxcywh_to_xyxy(x):
#     x_c, y_c, w, h = x.unbind(-1)
#     b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
#          (x_c + 0.5 * w), (y_c + 0.5 * h)]
#     return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    # 把负的 w、h 置 0
    w = w.clamp(min=0)
    h = h.clamp(min=0)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

# def box_xywh_to_xyxy(x):
#     x1, y1, w, h = x.unbind(-1)
#     b = [x1, y1, x1 + w, y1 + h]
#     return torch.stack(b, dim=-1)

def box_xywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    # x: [N,4] 格式 [x, y, w, h]
    x1, y1, w, h = x.unbind(-1)
    # 把负的 w、h 置 0
    w = w.clamp(min=0)
    h = h.clamp(min=0)
    b = [x1, y1, x1 + w, y1 + h]
    return torch.stack(b, dim=-1)

# def box_xywh_to_cxcywh(x):
#     x1, y1, w, h = x.unbind(-1)
#     b = [x1 + w/2, y1 + h/2, w, h]
#     return torch.stack(b, dim=-1)

def box_xywh_to_cxcywh(x):
    x1, y1, w, h = x.unbind(-1)
    w = w.clamp(min=0)
    h = h.clamp(min=0)
    b = [x1 + w/2, y1 + h/2, w, h]
    return torch.stack(b, dim=-1)

def box_xyxy_to_xywh(x):
    x1, y1, x2, y2 = x.unbind(-1)
    b = [x1, y1, x2 - x1, y2 - y1]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
'''Note that this function only supports shape (N,4)'''


def box_iou(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    area1 = box_area(boxes1) # (N,)
    area2 = box_area(boxes2) # (N,)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # (N,2)
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # (N,2)

    wh = (rb - lt).clamp(min=0)  # (N,2)
    inter = wh[:, 0] * wh[:, 1]  # (N,)

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


'''Note that this implementation is different from DETR's'''


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    boxes1: (N, 4)
    boxes2: (N, 4)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # try:
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2) # (N,)

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # (N,2)
    area = wh[:, 0] * wh[:, 1] # (N,)

    return iou - (area - union) / area, iou


def giou_loss(boxes1, boxes2):
    """

    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :return:
    """
    giou, iou = generalized_box_iou(boxes1, boxes2)
    return (1 - giou).mean(), iou

def giou_loss_match(boxes1, boxes2, match_map):
    """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :param match_map: (N), 0/1 mask
    :return: giou_loss, iou
    """
    mask = match_map == 1
    if mask.sum() == 0:
        return torch.tensor(0.0, device=boxes1.device), torch.tensor(0.0, device=boxes1.device)
    boxes1 = boxes1[mask]
    boxes2 = boxes2[mask]

    giou, iou = generalized_box_iou(boxes1, boxes2)
    giou_loss = (1 - giou).mean()

    return giou_loss, iou

def l1_loss_match(boxes1, boxes2, match_map):
    """
    :param boxes1: (N, 4) (x1,y1,x2,y2)
    :param boxes2: (N, 4) (x1,y1,x2,y2)
    :param match_map: (N), 0/1 mask
    :return: l1_loss
    """
    mask = match_map == 1
    if mask.sum() == 0:
        return torch.tensor(0.0, device=boxes1.device)
    boxes1 = boxes1[mask]
    boxes2 = boxes2[mask]

    loss = l1_loss(boxes1, boxes2, reduction="mean")

    return loss



def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    return [x1, y1, w, h]
