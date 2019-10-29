
#!/usr/bin/env python
# ------------------------------------------------------------------------
#
# Experimenting regression losses.
#
# Author Tuan Le (tuan.t.lei@gmail.com)
#
# ------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import warnings
import copy
import json
import numpy as np

import torch


class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(torch.log(torch.cosh(ey_t + 1e-12)))


class XTanhLoss(torch.nn.Module):
    def __init__(self):
        super(XTanhLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * torch.tanh(ey_t))


class XSigmoidLoss(torch.nn.Module):
    def __init__(self):
        super(XSigmoidLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        # return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)
        return torch.mean(2 * ey_t * torch.sigmoid(ey_t) - ey_t)


class AlgebraicLoss(torch.nn.Module):
    def __init__(self):
        super(AlgebraicLoss, self).__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(ey_t * ey_t / torch.sqrt(1 + ey_t * ey_t))


class CCCLoss(torch.nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()
    def forward(self, ground_truth, prediction):
        mean_gt = torch.mean (ground_truth, 0)
        mean_pred = torch.mean (prediction, 0)
        var_gt = torch.var(ground_truth, 0)
        var_pred = torch.var (prediction, 0)
        v_pred = prediction - mean_pred
        v_gt = ground_truth - mean_gt
        cor = torch.sum (v_pred * v_gt,0) / (torch.sqrt(torch.sum(v_pred ** 2,0)) * torch.sqrt(torch.sum(v_gt ** 2,0)))
        sd_gt = torch.std(ground_truth, 0)
        sd_pred = torch.std(prediction, 0)
        numerator=2*cor*sd_gt*sd_pred
        denominator=var_gt+var_pred+(mean_gt-mean_pred)**2
        ccc = numerator/denominator
        return 1-torch.mean(ccc)
