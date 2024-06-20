from __future__ import print_function

import os
import sys
import math
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform_1=None, transform_2=None):
        self.transform = [transform_1, transform_2]

    def __call__(self, idx, x):
        x_data = []
        for transform in self.transform:
            # augmentation
            if transform is not None:
                x_trans = transform(idx, x)
            else:
                x_trans = copy.deepcopy(x)
            
            # HWC -> CHW
            x_trans = x_trans.transpose((2,0,1))
            
            # float64 -> float32
            x_trans = x_trans.astype(np.float32)
            
            # numpy(float) -> tensor
            x_trans = torch.from_numpy(x_trans)
            
            x_data.append(x_trans)
            
        return x_data


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.NUM_EPOCHS)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def plot_confusion_matrix(save_dir, name_labels, y_true, y_pred, extra_msg=""):
    class_num = len(name_labels)
    id_labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=id_labels)
    cmx_data = pd.DataFrame(cmx_data, index=name_labels, columns=name_labels)
    if class_num <= 20:
        # output matplotlib format
        sns.heatmap(cmx_data, square=True, cbar=True, annot=True, fmt="d", cmap='OrRd', linewidths=0.5)
        plt.xlabel("predict", fontsize=8)
        plt.ylabel("ground truth", fontsize=8)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix{}.png'.format(extra_msg)), bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        # output pandas format
        cmx_data.to_csv(os.path.join(save_dir, 'confusion_matrix.csv'))
