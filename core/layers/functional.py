import math
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import time
from scipy.sparse.linalg import cg
from scipy.optimize import fmin_cg
from sklearn.mixture import GaussianMixture


# --------------------------------------
# pooling
# --------------------------------------

def mac(x):
    return F.max_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_max_pool2d(x, (1,1)) # alternative


def spoc(x):
    return F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    # return F.adaptive_avg_pool2d(x, (1,1)) # alternative


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)
    # return F.lp_pool2d(F.threshold(x, eps, eps), p, (x.size(-2), x.size(-1))) # alternative


def rmac(x, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    (tmp, idx) = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    v = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + eps).expand_as(v)

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b) - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b) - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                R = x[:, :, (int(i_) + torch.Tensor(range(wl)).long()).tolist(), :]
                R = R[:, :, :, (int(j_) + torch.Tensor(range(wl)).long()).tolist()]
                vt = F.max_pool2d(R, (R.size(-2), R.size(-1)))
                vt = vt / (torch.norm(vt, p=2, dim=1, keepdim=True) + eps).expand_as(vt)
                v += vt

    return v


def roipool(x, rpool, L=3, eps=1e-6):
    ovr = 0.4  # desired overlap of neighboring regions
    steps = torch.Tensor([2, 3, 4, 5, 6, 7])  # possible regions for the long dimension

    W = x.size(3)
    H = x.size(2)

    w = min(W, H)
    w2 = math.floor(w / 2.0 - 1)

    b = (max(H, W) - w) / (steps - 1)
    _, idx = torch.min(torch.abs(((w ** 2 - w * b) / w ** 2) - ovr), 0)  # steps(idx) regions for long dimension

    # region overplus per dimension
    Wd = 0;
    Hd = 0;
    if H < W:
        Wd = idx.item() + 1
    elif H > W:
        Hd = idx.item() + 1

    vecs = []
    vecs.append(rpool(x).unsqueeze(1))

    for l in range(1, L + 1):
        wl = math.floor(2 * w / (l + 1))
        wl2 = math.floor(wl / 2 - 1)

        if l + Wd == 1:
            b = 0
        else:
            b = (W - wl) / (l + Wd - 1)
        cenW = torch.floor(wl2 + torch.Tensor(range(l - 1 + Wd + 1)) * b).int() - wl2  # center coordinates
        if l + Hd == 1:
            b = 0
        else:
            b = (H - wl) / (l + Hd - 1)
        cenH = torch.floor(wl2 + torch.Tensor(range(l - 1 + Hd + 1)) * b).int() - wl2  # center coordinates

        for i_ in cenH.tolist():
            for j_ in cenW.tolist():
                if wl == 0:
                    continue
                vecs.append(rpool(x.narrow(2, i_, wl).narrow(3, j_, wl)).unsqueeze(1))

    return torch.cat(vecs, dim=1)


# --------------------------------------
# normalization
# --------------------------------------

def l2n(x, eps=1e-6):
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)


def powerlaw(x, eps=1e-6):
    x = x + eps
    return x.abs().sqrt().mul(x.sign())


# --------------------------------------
# loss
# --------------------------------------

def contrastive_loss(x, label, margin=0.7, eps=1e-6):
    # x is D x N
    dim = x.size(0)  # D
    nq = torch.sum(label.data == -1)  # number of tuples
    S = x.size(1) // nq  # number of images per tuple including query: 1+1+n

    x1 = x[:, ::S].permute(1, 0).repeat(1, S - 1).view((S - 1) * nq, dim).permute(1, 0)
    idx = [i for i in range(len(label)) if label.data[i] != -1]
    x2 = x[:, idx]
    lbl = label[label != -1]

    dif = x1 - x2
    D = torch.pow(dif + eps, 2).sum(dim=0).sqrt()

    y = 0.5 * lbl * torch.pow(D, 2) + 0.5 * (1 - lbl) * torch.pow(torch.clamp(margin - D, min=0), 2)
    y = torch.sum(y)
    return y


def gmp(matrix, lamb=10, cgd=False):
    """
    matrix is D*N
    return the gmp vector by add the
    :param cgd: whether use Conjugate Gradient Descent method
    :param matrix:
    :param lamb: need cross valid(10, 100, 1000, 10000)
    :return:
    """
    D, N = matrix.shape
    if not cgd:
        inv_matrix = np.linalg.inv((np.dot(matrix, np.transpose(matrix)) + lamb * np.diag([1] * D)).astype(np.float32))
    else:
        inv_matrix = cg(np.dot(matrix, np.transpose(matrix)) + lamb * np.diag([1] * D), np.diag([1] * D))

    gmp = np.dot(np.dot(inv_matrix, matrix), np.array([1] * N).T)

    return gmp


def gmm(matrix):
    """
    :param matrix: N*C samples
    :return:
    """
    gmm4 = GaussianMixture(n_components=4, covariance_type='full').fit(matrix)
    return gmm4.means_


def create_mean(data):
    """
    data is N*W*H*C

    :param data:
    :return:
    """
    nfeats = 0
    dim = data[0].shape[2]
    mean_value = np.zeros(dim)
    for i in range(data.shape[0]):
        ele = data[i]
        ele = ele.reshape(-1, dim)
        ele = ele.T
        mean_value += np.sum(ele, axis=1)
        nfeats += 1
    return mean_value / nfeats


def weight_Heat(feature_map, mean):
    """

    :param feature_map: W*H*C
    :param mean: a vector of C length
    :return:
    """
    w, h, c = feature_map.shape
    s = np.sum(feature_map, axis=2)
    feature_map = feature_map.reshape(-1, c)
    feature_map = feature_map.T
    feature_map_org = feature_map.copy()
    feature_map = feature_map - np.expand_dims(mean, axis=1).repeat(feature_map.shape[1], axis=1)
    from sklearn.preprocessing import normalize
    feature_map = normalize(feature_map, axis=0)
    s0 = s.reshape(-1, 1)
    a = np.dot(feature_map.T, feature_map)
    a = a + np.diag(-np.diag(a))
    a = np.where(a > 0.1, a, 0)
    const_z = 0.1
    z = const_z * np.mean(a[a > 0])
    weights = get_potential_inv(a, z)
    weights = 1. / weights
    weighted_feature_map = feature_map_org * weights
    return np.sum(weighted_feature_map, axis=1)


def get_potential_inv(a, z):
    """
    A IS THE AFFINIT MATRIX
    :param a:
    :param z:
    :return:
    """
    lamb = z * np.ones(a.shape[0])
    a = np.concatenate((a, np.expand_dims(lamb, axis=1)), axis=1)
    a = np.concatenate((a, np.expand_dims(np.append(np.ones(a.shape[0]), 0.0), axis=0)), axis=0)

    sa = np.sum(a, axis=1)
    a = a / sa[:, None]
    lap_mat = np.diag(np.sum(a, axis=1)) - a
    lap_mat = lap_mat[:-1, :-1]

    lap_mat_inv = np.linalg.inv(lap_mat)

    deno = np.diag(lap_mat_inv)
    lap_mat_inv = lap_mat_inv - np.diag(deno)
    reward = np.sum(lap_mat_inv, axis=0) / deno
    return reward / a.shape[0]


def get_potential_inv_re(a, z):
    lamb = z * np.ones(a.shape[0])
    a = np.concatenate((a, np.expand_dims(lamb, axis=1)), axis=1)
    a = np.concatenate((a, np.expand_dims(np.append(np.ones(a.shape[0]), 0.0), axis=0)), axis=0)

    sa = np.sum(a, axis=1)
    a = a / sa[:, None]

    lap_mat = np.diag(np.sum(a, axis=1)) - a
    lap_mat = lap_mat[:-1, :-1]
    lap_mat_inv = np.linalg.inv(lap_mat)
    return lap_mat_inv[:-1, -1]


if __name__ == '__main__':
    m1 = np.random.randn(512, 20).astype(np.float32)
    gmp1 = gmp(m1.copy(), lamb=10, cgd=False)
    # gmp2 = gmp(m1.copy(), lamb=10, cgd=True)
    # print(gmp1)
    # print(gmp2)

    # heat diffusion

    weight_Heat(np.random.randn(5, 5, 16), np.random.randn(16))
