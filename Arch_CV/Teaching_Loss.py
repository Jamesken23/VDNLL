"""
Ref: https://github.com/bhanML/Co-teaching/blob/master/loss.py
Ref: https://github.com/xingruiyu/coteaching_plus/blob/master/loss.py
"""

import torch, scipy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-12


# Co-Teaching的损失函数，forget_rate 指舍弃多少比例的样本，其余样本可被视为是干净的
def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduce=False)
    ind_1_sorted = np.argsort(loss_1.data.cpu())
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce=False)
    ind_2_sorted = np.argsort(loss_2.data.cpu())
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # 交换更新的标签信息，进行协同训练
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    if torch.isnan(loss_1_update):
        print(loss_1_update, y_1[ind_2_update], t[ind_2_update], len(loss_1_sorted), remember_rate)

    return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember



def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)


# JoCR的损失函数
def loss_jocor(y1, y2, labels, forget_rate, co_lambda=0.1):
    loss_pick_1 = F.cross_entropy(y1, labels, reduce = False) * (1 - co_lambda)
    loss_pick_2 = F.cross_entropy(y2, labels, reduce = False) * (1 - co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y1, y2,reduce=False) + co_lambda * kl_loss_compute(y2, y1, reduce=False)).cpu()

    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss = torch.mean(loss_pick[ind_update])

    return loss, loss


def Distance_squared(x, y, featdim=1):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    d = dist.clamp(min=eps)
    d[torch.eye(d.shape[0]) == 1] = eps
    return d

def CalPairwise(dist):
    dist[dist < 0] = 0
    Pij = torch.exp(-dist)
    return Pij

def CalPairwise_t(dist, v):
    C = scipy.special.gamma((v + 1) / 2) / (np.sqrt(v * np.pi) * scipy.special.gamma(v / 2))
    return torch.pow((1 + torch.pow(dist, 2) / v), - (v + 1) / 2)

def CE(P, Q):
    loss = -1 * (P * torch.log(Q + eps)).mean()
    return loss


def loss_structrue(feat1, feat2):
    q1 = CalPairwise(Distance_squared(feat1, feat1))
    q2 = CalPairwise(Distance_squared(feat2, feat2))
    return -1 * (q1 * torch.log(q2 + eps)).mean()


def loss_structrue_t(feat1, feat2, v):
    q1 = CalPairwise_t(Distance_squared(feat1, feat1), v)
    q2 = CalPairwise_t(Distance_squared(feat2, feat2), v)
    return -1 * (q1 * torch.log(q2 + eps)).mean()