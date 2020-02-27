import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import core.layers.functional as LF
from core.layers.normalization import L2N
import joblib
from sklearn.preprocessing import normalize
import numpy as np


# --------------------------------------
# Pooling layers
# --------------------------------------

class MAC(nn.Module):

    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x):
        return LF.mac(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class SPoC(nn.Module):

    def __init__(self):
        super(SPoC, self).__init__()

    def forward(self, x):
        return LF.spoc(x)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return LF.gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class RMAC(nn.Module):

    def __init__(self, L=3, eps=1e-6):
        super(RMAC, self).__init__()
        self.L = L
        self.eps = eps

    def forward(self, x):
        return LF.rmac(x, L=self.L, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'L=' + '{}'.format(self.L) + ')'


class Hew(nn.Module):
    def __init__(self):
        super(Hew, self).__init__()

    def forward(self, x):
        mean = joblib.load('hew_means.pkl')
        x = x.detach().cpu().numpy()
        return torch.Tensor(LF.weight_Heat(x, mean)).float()


class Rpool(nn.Module):

    def __init__(self, rpool, whiten=None, L=3, eps=1e-6):
        super(Rpool, self).__init__()
        self.rpool = rpool
        self.L = L
        self.whiten = whiten
        self.norm = L2N()
        self.eps = eps

    def forward(self, x, aggregate=True):
        # features -> roipool
        o = LF.roipool(x, self.rpool, self.L, self.eps)  # size: #im, #reg, D, 1, 1
        # concatenate regions from all images in the batch
        s = o.size()
        o = o.view(s[0] * s[1], s[2], s[3], s[4])  # size: #im x #reg, D, 1, 1
        # rvecs -> norm
        o = self.norm(o)
        # rvecs -> whiten -> norm
        if self.whiten is not None:
            o = self.norm(self.whiten(o.squeeze(-1).squeeze(-1)))
        # reshape back to regions per image
        o = o.view(s[0], s[1], s[2], s[3], s[4])  # size: #im, #reg, D, 1, 1
        # aggregate regions into a single global vector per image
        o = torch.squeeze(o, 0)
        o = torch.squeeze(o, -1)
        o = torch.squeeze(o, -1)
        if aggregate == 'sum':
            # rvecs -> sumpool -> norm
            o = normalize(np.sum(o.detach().cpu().numpy(), axis=0).reshape(1, -1))
        elif aggregate == 'gmm':
            o = LF.gmm(o.detach().cpu().numpy())
            o = normalize(o)
        elif aggregate == 'gmp':
            o = LF.gmp(o.detach().cpu().numpy().astype(np.float32).T)
            o = normalize(o.reshape(1, -1))
        return o

    def __repr__(self):
        return super(Rpool, self).__repr__() + '(' + 'L=' + '{}'.format(self.L) + ')'
