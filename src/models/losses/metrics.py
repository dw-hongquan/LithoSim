'''
Author: Hongquan
Date: Apr. 15, 2025
'''
import torch
from torch import Tensor
from typing import Tuple
import cv2
import torch.nn.functional as F

def iou_dice(pred: Tensor, target: Tensor, smooth: float = 1e-5, activate: bool = True) -> Tuple[float, float]:
    '''
    pred: predicted resist [batch_size, 1, 4096, 4096]
    target: ground truth resist [batch_size, 1, 4096, 4096]
    return: iou and dice coff.
    '''
    if activate:
        output_data = torch.sigmoid(pred).data.cpu().numpy()
    else:
        output_data = pred.data.cpu().numpy()
    target_data = target.data.cpu().numpy()

    output_ = output_data > 0.5
    target_ = target_data > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    iou: float = (intersection + smooth) / (union + smooth)
    dice: float = (2 * iou) / (iou + 1)

    return iou, dice

def cal_acc(pred: Tensor, target: Tensor, activate: bool = True) -> float:
    '''
    pred: predicted resist [batch_size, 1, 4096, 4096]
    target: ground truth resist [batch_size, 1, 4096, 4096]
    return: binary acc.
    '''
    b, c, h, w = target.shape
    if activate:
        output_data = torch.sigmoid(pred)
    else:
        output_data = pred

    output_ = output_data > 0.5
    target_ = target > 0.5

    return 1 - torch.logical_xor(output_, target_).data.cpu().numpy().sum() / (b * c * h * w)

def cal_mse(pred: Tensor, target: Tensor, activate: True) -> float:
    """
    MSE calculation
    
    参数:
        pred (torch.Tensor): 预测值，形状为 [B, C, H, W]
        target (torch.Tensor): 目标值，形状为 [B, C, H, W]
        
    返回:
        float: MSE 值
    """
    assert pred.shape == target.shape, f"pred and target shape must be same, but got pred : {pred.shape}, target: {target.shape}"
    if activate:
        pred_data = torch.where(torch.sigmoid(pred) > 0.5, 1.0, 0.0).float()
    else:
        pred_data = torch.where(pred > 0.5, 1.0, 0.0).float()

    mse = F.mse_loss(pred_data, target, reduction='mean')
    
    return mse.item()

class EPECalculator:
    def __init__(self, epe_spacing=30, min_edge_length=60, corner_rounding=80, max_epe=100):
        self.EPE_SPACING = epe_spacing
        self.MIN_EDGE_LENGTH = min_edge_length
        self.CORNER_ROUNDING = corner_rounding
        self.MAX_EPE = max_epe
        try:
            self.epe = self.calculate_epe()
        except IndexError:
            self.epe = 20

    def _find_nearest_edge(self, image, point, direction):
        min_distance = 2048

        if direction == 'right':
            temp = point.copy()
            up_dis = 0
            down_dis = 0
            temp[1] += 1
            while temp[1] < image.shape[0] and image[temp[1]][temp[0]] == image[point[1]][point[0]] and up_dis < self.MAX_EPE:
                temp[1] += 1
                up_dis += 1
            temp = point.copy()
            temp[1] -= 1
            while temp[1] >= 0 and image[temp[1]][temp[0]] == image[point[1]][point[0]] and down_dis < self.MAX_EPE:
                temp[1] -= 1
                down_dis += 1
            moving = min(up_dis, down_dis)
            min_distance = min(moving, min_distance)

        elif direction == 'left':
            temp = point.copy()
            up_dis = 0
            down_dis = 0
            temp[1] += 1
            while temp[1] < image.shape[0] and image[temp[1]][temp[0]] == image[point[1]][point[0]] and up_dis < self.MAX_EPE:
                temp[1] += 1
                up_dis += 1
            temp = point.copy()
            temp[1] -= 1
            while temp[1] >= 0 and image[temp[1]][temp[0]] == image[point[1]][point[0]] and down_dis < self.MAX_EPE:
                temp[1] -= 1
                down_dis += 1
            moving = min(up_dis, down_dis)
            min_distance = min(moving, min_distance)

        elif direction == 'up':
            temp = point.copy()
            left_dis = 0
            right_dis = 0
            temp[0] += 1
            while temp[0] < image.shape[1] and image[temp[1]][temp[0]] == image[point[1]][point[0]] and right_dis < self.MAX_EPE:
                temp[0] += 1
                right_dis += 1
            temp = point.copy()
            temp[0] -= 1
            while temp[0] >= 0 and image[temp[1]][temp[0]] == image[point[1]][point[0]] and left_dis < self.MAX_EPE:
                temp[0] -= 1
                left_dis += 1
            moving = min(left_dis, right_dis)
            min_distance = min(moving, min_distance)

        elif direction == 'down':
            temp = point.copy()
            left_dis = 0
            right_dis = 0
            temp[0] += 1
            while temp[0] < image.shape[1] and image[temp[1]][temp[0]] == image[point[1]][point[0]] and right_dis < self.MAX_EPE:
                temp[0] += 1
                right_dis += 1
            temp = point.copy()
            temp[0] -= 1
            while temp[0] >= 0 and image[temp[1]][temp[0]] == image[point[1]][point[0]] and left_dis < self.MAX_EPE:
                temp[0] -= 1
                left_dis += 1
            moving = min(left_dis, right_dis)
            min_distance = min(moving, min_distance)

        return min_distance

    def _calculate_edge_placement_errors(self, result_image_pred, result_image_gold, points, dirs):
        errors = 0
        for p in range(len(points)):
            epe1 = self._find_nearest_edge(result_image_pred, points[p], dirs[p])
            epe2 = self._find_nearest_edge(result_image_gold, points[p], dirs[p])
            errors += abs(epe1 - epe2)
        return errors

    def _add_points(self, points, dirs, initial_point, end_point, direction):
        if direction == "right":
            length = end_point[0] - initial_point[0]
            if length < self.MIN_EDGE_LENGTH:
                points.append([initial_point[0] + int(length/2), initial_point[1]])
                dirs.append(direction)
            else:
                remainder = (length - self.CORNER_ROUNDING) % self.EPE_SPACING
                corner = int(self.CORNER_ROUNDING / 2 + remainder / 2)
                x_label = end_point[0] - corner
                current_x = initial_point[0] + corner
                points.append([current_x, initial_point[1]])
                dirs.append(direction)
                while current_x + self.EPE_SPACING <= x_label:
                    current_x += self.EPE_SPACING
                    points.append([current_x, initial_point[1]])
                    dirs.append(direction)

        elif direction == "left":
            length = initial_point[0] - end_point[0]
            if length < self.MIN_EDGE_LENGTH:
                points.append([initial_point[0] - int(length/2), initial_point[1]])
                dirs.append(direction)
            else:
                remainder = (length - self.CORNER_ROUNDING) % self.EPE_SPACING
                corner = int(self.CORNER_ROUNDING / 2 + remainder / 2)
                x_label = end_point[0] + corner
                current_x = initial_point[0] - corner
                points.append([current_x, initial_point[1]])
                dirs.append(direction)
                while current_x - self.EPE_SPACING >= x_label:
                    current_x -= self.EPE_SPACING
                    points.append([current_x, initial_point[1]])
                    dirs.append(direction)

        elif direction == "up":
            length = end_point[1] - initial_point[1]
            if length < self.MIN_EDGE_LENGTH:
                points.append([initial_point[0], initial_point[1] + int(length/2)])
                dirs.append(direction)
            else:
                remainder = (length - self.CORNER_ROUNDING) % self.EPE_SPACING
                corner = int(self.CORNER_ROUNDING / 2 + remainder / 2)
                y_label = end_point[1] - corner
                current_y = initial_point[1] + corner
                points.append([initial_point[0], current_y])
                dirs.append(direction)
                while current_y + self.EPE_SPACING <= y_label:
                    current_y += self.EPE_SPACING
                    points.append([initial_point[0], current_y])
                    dirs.append(direction)

        elif direction == "down":
            length = initial_point[1] - end_point[1]
            if length < self.MIN_EDGE_LENGTH:
                points.append([initial_point[0], initial_point[1] - int(length/2)])
                dirs.append(direction)
            else:
                remainder = (length - self.CORNER_ROUNDING) % self.EPE_SPACING
                corner = int(self.CORNER_ROUNDING / 2 + remainder / 2)
                y_label = end_point[1] + corner
                current_y = initial_point[1] - corner
                points.append([initial_point[0], current_y])
                dirs.append(direction)
                while current_y - self.EPE_SPACING >= y_label:
                    current_y -= self.EPE_SPACING
                    points.append([initial_point[0], current_y])
                    dirs.append(direction)

    def calculate_epe(self, target_path: str, pred_path: str, gold_path: str) -> float:
        '''
        Since EPE calculation needs layout file, make sure predicted resist is binarized and saved to pred_path.
        pred_path: predicted resist path
        target_path: ground truth resist path
        layout_path: corresponding layout file path
        return: edge placement error.
        '''
        target_image = cv2.imread(target_path, 0)
        result_image_pred = cv2.imread(pred_path, 0)
        result_image_gold = cv2.imread(gold_path, 0)

        contours, _ = cv2.findContours(target_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = []
        dirs = []
        for contour in contours:
            for i in range(len(contour)):
                if i == len(contour) - 1:
                    dir_vector = (contour[0] - contour[i])[0]
                    ip = contour[i][0]
                    ep = contour[0][0]
                else:
                    dir_vector = (contour[i+1] - contour[i])[0]
                    ip = contour[i][0]
                    ep = contour[i+1][0]

                if (dir_vector[0]**2 + dir_vector[1]**2) <= 5:
                    continue

                if dir_vector[0] == 0:
                    direction = "up" if dir_vector[1] > 0 else "down"
                else:
                    direction = "right" if dir_vector[0] > 0 else "left"

                self._add_points(points, dirs, ip, ep, direction)

        if not points:
            return 0.0

        errors = self._calculate_edge_placement_errors(result_image_pred, result_image_gold, points, dirs)
        epe = errors / len(points)
        return epe

