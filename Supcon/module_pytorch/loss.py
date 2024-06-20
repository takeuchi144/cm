#!/usr/bin/env python
# coding:utf-8
import torch

# SEPARATE_ABS_SIGN = 3 (class_num)
class CCEPlusMSELoss(torch.nn.Module):
    def __init__(self, device, num_classes, offset, epsilon=1e-7):
        super().__init__()
        self.epsilon = torch.tensor(epsilon)
        arange_arr = torch.arange(num_classes)
        arange_arr -= offset
        arange_arr = arange_arr.to(torch.float32)
        weight_arr = torch.square(arange_arr)
        weight_arr /= torch.max(weight_arr)
        weight_arr += 0.5
        self.weight_arr = weight_arr.to(device)

    def forward(self, y_pred, y_true):
        cdf_y_true = torch.clip(torch.cumsum(y_true, axis=-1), 0, 1) / self.weight_arr
        cdf_y_pred = torch.clip(torch.cumsum(y_pred, axis=-1), 0, 1) / self.weight_arr
        loss = torch.mean(torch.sqrt(torch.maximum(torch.mean(torch.square(cdf_y_true - cdf_y_pred), axis=[-1, -2]), self.epsilon)))
        return loss

# SEPARATE_ABS_SIGN = 1 (concat(1,1))
class DistSignLoss(torch.nn.Module):
    def __init__(self, loss_weight):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, y_pred, y_true):
        y_true_abs = y_true[:,0]
        y_pred_abs = y_pred[:,0]
        y_true_sign = y_true[:,1]
        y_pred_sign = y_pred[:,1]
        loss = (y_true_abs - y_pred_abs)**2 - self.loss_weight * (y_true_sign * torch.log(y_pred_sign + 1.0e-6) + (1.0 - y_true_sign) * torch.log(1.0 - y_pred_sign + 1.0e-6))
        loss = torch.mean(loss)
        return loss

# SEPARATE_ABS_SIGN = 1 (concat(1,1))
class MSETotLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_true_abs = y_true[:,0]
        y_pred_abs = y_pred[:,0]
        y_true_sign = y_true[:,1]*2.0-1.0
        y_pred_sign = y_pred[:,1]*2.0-1.0
        loss = torch.sqrt((y_true_abs*torch.sign(y_true_sign) - y_pred_abs*torch.sign(y_pred_sign))**2)
        loss = torch.mean(loss)
        return loss

# SEPARATE_ABS_SIGN = 2 (1)
class WeightedMSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.loss_func = torch.nn.MSELoss(reduction=reduction) # "mean", "sum", "none"

    def forward(self, y_pred, y_true):
        loss = self.loss_func(y_true, y_pred) # / (1.0 + torch.abs(y_true))
        return loss


