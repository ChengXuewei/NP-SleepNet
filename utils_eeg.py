import numpy as np
from sleepstage import class_dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_seq_ids(fname):
    """Load sequence of IDs from txt file."""
    ids = []
    with open(fname, "r") as f:
        for line in f:
            ids.append(int(line.strip()))
    ids = np.asarray(ids)
    return ids

def print_n_samples_each_class(labels):
    """Print the number of samples in each class."""

    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(class_dict[c], n_samples))


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss 实现
        :param alpha: 类别权重 (Tensor)，用于处理类别不平衡
        :param gamma: 调节因子，γ≥0，越大对难样本关注越高
        :param reduction: 'none' | 'mean' | 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 计算 p_t
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


