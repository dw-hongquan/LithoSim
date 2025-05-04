import sys
sys.path.append('.')

import torch
import time
from torch import Tensor, device
from src.models.litho_module import LithoLitModule
import torchvision.utils as U
from src.models.losses.metrics import iou_dice, cal_acc, cal_mse
import cv2
import os
from typing import Tuple

def src2tensor(src_path: str, device: device) -> Tensor:
    data = []
    with open(src_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            row = [float(parts[0]), float(parts[1]), float(parts[2])]
            data.append(row)

    return torch.tensor(data, device=device).unsqueeze(0)


def litho_eval(lithosim_path: str,
               src_path: str,
               mask_path: str,
               dose: float = 0,
               defocus: float = 0,
               resist_path: str = '',
               cuda_id: device = 'cpu',
               activate: bool = True,
               vis: bool = True,
               save_path: str = '') -> Tuple[float, float, float, float]:
    src = src2tensor(src_path, cuda_id)
    mask = torch.tensor(cv2.imread(mask_path, 0) / 255, dtype=torch.float32, device=cuda_id).unsqueeze(0).unsqueeze(0)
    dose_t = torch.tensor(dose, dtype=torch.float32, device=cuda_id).unsqueeze(0).unsqueeze(0)
    defocus_t = torch.tensor(defocus, dtype=torch.float32, device=cuda_id).unsqueeze(0).unsqueeze(0)
    lithosim = LithoLitModule.load_from_checkpoint(checkpoint_path = lithosim_path, map_location = cuda_id)
    time_start = time.monotonic()
    resist_pred = lithosim(src, mask, dose_t, defocus_t)
    time_end = time.monotonic()
    resist_gt = torch.tensor(cv2.imread(resist_path, 0) / 255, dtype=torch.float32, device=cuda_id).unsqueeze(0).unsqueeze(0)
    iou, _ = iou_dice(resist_pred, resist_gt, activate=activate)
    pa = cal_acc(resist_pred, resist_gt, activate=activate)
    mse = cal_mse(resist_pred, resist_gt, activate=activate)
    tat = time_end - time_start
    if vis:
        visual(resist_pred, save_path, activate)
    return iou, pa, mse, tat

def visual(pred: Tensor, save_path: str = '', activate: bool = True):
    if activate:
        pred_data = torch.where((torch.sigmoid(pred) > 0.5), 1.0, 0.0).float()
    else:
        pred_data = torch.where(pred > 0.5, 1.0, 0.0).float()
    U.save_image(pred_data, save_path)

if __name__ == '__main__':
    lithosim_path = '/home/hehq/src/hongquan-gen/logs/doinn-fused/runs/2025-04-23_19-18-45/checkpoints/last.ckpt'
    test_dir = '/home/hehq/dataset/lithosim/opc_metal/test/50001'
    src_path = os.path.join(test_dir, 'source_simple.src')
    mask_path = os.path.join(test_dir, 'mask.png')
    resist_path = '/home/hehq/dataset/lithosim/opc_metal/test/50001/RI_dose_0_defocus_0.png'
    save_path = '/home/hehq/dataset/visual/1.png'
    iou, pa, mse, tat = litho_eval(lithosim_path, src_path, mask_path, resist_path=resist_path, save_path=save_path)
    print(iou, pa, mse, tat)
