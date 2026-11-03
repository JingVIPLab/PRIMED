import torch

import os
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def count_parameters(model):
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = total - trainable
    buffers = sum(buffer.numel() for buffer in model.buffers())
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "buffers": buffers,
    }


def format_parameter_count(count):
    return f"{count:,} ({count / 1e6:.2f}M)"


def log_parameter_count(model, prefix="Model"):
    stats = count_parameters(model)
    message = (
        f"{prefix} Parameters | "
        f"total: {format_parameter_count(stats['total'])}, "
        f"trainable: {format_parameter_count(stats['trainable'])}, "
        f"frozen: {format_parameter_count(stats['frozen'])}, "
        f"buffers: {format_parameter_count(stats['buffers'])}"
    )
    logging.info(message)
    print(message)
    return stats
                
def metric_s_for_null(pred):
    assert len(pred.shape) == 3

    N = pred.size(0)
    num_pixels = pred.view(-1).shape[0] 

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()

    x = torch.sum(pred.view(-1)) 
    s = torch.sqrt(x / num_pixels)

    return s


def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param: 
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.5).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N

    return iou


def _eval_pr(y_pred, y, num, cuda_flag=True):
    # Get device from input tensors to ensure consistency
    device = y_pred.device
    
    prec = torch.zeros(num, device=device)
    recall = torch.zeros(num, device=device)
    thlist = torch.linspace(0, 1 - 1e-10, num, device=device)
    
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall


def Eval_Fmeasure(pred, gt, measure_path=None, pr_num=255):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # print('=> eval [FMeasure]..')
    # pred = torch.sigmoid(pred) # =======================================[important]
    N = pred.size(0)
    beta2 = 0.3
    avg_f, img_num = 0.0, 0
    # Get device from input tensor
    device = pred.device
    score = torch.zeros(pr_num, device=device)
    # fLog = open(os.path.join(measure_path, 'FMeasure.txt'), 'w')
    # print("{} videos in this batch".format(N))

    for img_id in range(N):
        # examples with totally black GTs are out of consideration
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0 # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num
        # print('score: ', score)
    # fLog.close()

    return score.max().item()


def save_mask(pred_masks, save_base_path, video_name_list):
    # pred_mask: [bs*5, 1, 224, 224]
    # print(f"=> {len(video_name_list)} videos in this batch")

    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path, exist_ok=True)

    pred_masks = pred_masks.squeeze(2)
    pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
    
    pred_masks = pred_masks.view(-1, 5, pred_masks.shape[-2], pred_masks.shape[-1])
    pred_masks = pred_masks.cpu().data.numpy().astype(np.uint8)
    pred_masks *= 255
    bs = pred_masks.shape[0]

    for idx in range(bs):
        video_name = video_name_list[idx]
        mask_save_path = os.path.join(save_base_path, video_name)
        if not os.path.exists(mask_save_path):
            os.makedirs(mask_save_path, exist_ok=True)
        one_video_masks = pred_masks[idx] # [5, 1, 224, 224]
        for video_id in range(len(one_video_masks)):
            one_mask = one_video_masks[video_id]
            output_name = "%s_%d.png"%(video_name, video_id)
            im = Image.fromarray(one_mask).convert('P')
            im.save(os.path.join(mask_save_path, output_name), format='PNG')


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v
