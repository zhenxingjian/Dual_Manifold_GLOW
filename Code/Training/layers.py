# In this script, the function should be specified for model, so that model will be lean
from __future__ import division

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

import utils


class CompSensingDataset(Dataset):
    def __init__(self, X, Y):
        super(CompSensingDataset, self).__init__()
        self.X = torch.from_numpy(X).float()  # [1000, 32, 32, 32, 3, 3]
        self.Y = torch.from_numpy(Y).float()  # [1000, 32, 32, 32, 3, 3]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        x = self.X[item]  # [32, 32, 32, 3, 3]
        y = self.Y[item]  # [32, 32, 32, 3, 3]
        return x, y


class DepthSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthSepConv, self).__init__()
        self.depth = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.channel = nn.Conv3d(in_channels, out_channels, kernel_size=1, groups=1)

    def forward(self, x):
        out = self.depth(x)
        out = self.channel(out)
        return out


def gaussian_spd(mean, std=1):
    """
    function for generating a set of functions corresponding to the gaussian distribution defined on SPD given mean and
    std, mean in SPD(3) while std is scalar
    NOTE no need for sampling so far ...
    :param mean: tensor shape [1, 1, 1, 1, 1, 3, 3], assume voxel-wise same
    :param std: scalar, default to be 1
    :return: a set of method ...
    """
    #assert_spd(mean)

    class o(object):
        pass

    o.mean = mean
    o.std = std
    o.logps = lambda x: - spdist_wrapper(x, mean) / (2 * std * std)
    o.logp = lambda x: o.logps(x).sum(dim=(1, 2, 3, 4))
    return o


def gaussian_real(mean, logsd, dim):
    """
    function for generating a set of function corresponding to the gaussian distribution defined by mean and log(std)
    :param mean: tensor shape [batch, C, m, m, m, 6]
    :param logsd: tensor shape [batch, C, m, m, m, 6]
    :return: a set of method ...
    """
    B, C, m = assert_real(mean, dim)
    assert mean.shape == logsd.shape

    class o(object):
        pass

    o.mean = mean
    o.logstd = logsd
    # logcov = torch.zeros(B, C, m, m, m, dim, dim).to(logsd.device)
    # covinv = torch.zeros(B, C, m, m, m, dim, dim).to(logsd.device)
    # for d in range(dim):
    #     # logcov[..., d, d] = logsd[..., d]
    #     # covinv[..., d, d] = 1 / torch.exp(logsd[..., d])
    # o.logcov = logcov
    stdinv = 1 / torch.exp(logsd)  # [B, C, m, m, m, 6]
    o.eps = torch.randn(mean.shape)
    o.sample = mean + torch.exp(logsd) * o.eps.to(mean.device)
    o.sample2 = lambda eps: mean + torch.exp(logsd) * eps.to(mean.device)

    nlog2pi = dim * np.log(2 * np.pi)
    logcov = logsd.sum(dim=-1).unsqueeze(-1)  # [B, C, m, m, m, 1]
    #centerx = (x - mean).unsqueeze(-2)  # [B, C, m, m, m, 1, 6]
    #centerxT = (x - mean).unsqueeze(-1)  # [B, C, m, m, m, 6, 1]
    #xsquare = centerx @ covinv  # [B, C, m, m, m, 1, 6]
    #xsquare = xsquare @ centerxT  # [B, C, m, m, m, 1, 1]
    #xsquare = xsquare.squeeze(-1)  # [B, C, m, m, m, 1]
    o.logps = lambda x: -0.5 * (nlog2pi + logcov + (x - mean) ** 2 * stdinv)
    #o.logps = lambda x: -0.5 * ((x - mean) ** 2 * stdinv)
    # o.logps = lambda x: -0.5 * (dim * np.log(2 * np.pi) + logsd.sum(dim=-1).unsqueeze(-1) + (x - mean) ** 2 / torch.exp(2. * logsd))
    o.logp = lambda x: o.logps(x).sum(dim=(1, 2, 3, 4))
    # o.get_eps = lambda x: (x - mean) / torch.exp(logsd)
    return o


def spdist_wrapper(X, M):
    """
    this function wrap utils.spdist, but with specified tensor size
    :param X: [..., 3, 3]
    :param Y: [..., 3, 3]
    :return: [...]
    """
    B, C, m = assert_spd(X)
    M = M.expand(B, C, m, m, m, -1, -1)
    X = X.view(-1, 3, 3)
    M = M.view(-1, 3, 3)
    Z = utils.spdist(X, M)  # [N]
    Z = Z.view(B, C, m, m, m)
    return Z


def squeeze2d(x, factor=2):
    """
    function for squeeze method
    :param x: tensor [batch, C, m, m, d, d]
    :param factor: default 2, how much squeezed
    :return: tensor [batch, C * factor * factor, m // factor, m // factor, d, d]
    """
    assert factor >= 1
    if factor == 1:
        return x
    if len(x.shape) == 6:
        [batch, C, m, _, d, _] = x.shape
        assert x.shape[-1] == d
    elif len(x.shape) == 5:
        [batch, C, m, _, d] = x.shape
    else:
        raise NotImplementedError
    assert x.shape[3] == m
    assert m % factor == 0
    if len(x.shape) == 6:
        x = x.reshape(batch, C, m // factor, factor, m // factor, factor, d, d)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
        # [batch, C, factor, factor, m // factor, m // factor, d, d]
        x = x.contiguous().view(batch, C * factor * factor, m // factor, m // factor, d, d)
    elif len(x.shape) == 5:
        x = x.reshape(batch, C, m // factor, factor, m // factor, factor, d)
        x = x.permute(0, 1, 3, 5, 2, 4, 6)
        # [batch, C, factor, factor, m // factor, m // factor, d]
        x = x.contiguous().view(batch, C * factor * factor, m // factor, m // factor, d)
    else:
        raise NotImplementedError
    return x


def unsqueeze2d(x, factor=2):
    """
    function for squeeze back
    :param x: tensor [batch, C * factor * factor * factor, m // factor, m // factor, m // factor, d, d]
    :param factor: default 2
    :return: tensor [batch, C, m, m, d, d]
    """
    assert factor >= 1
    if factor == 1:
        return x
    if len(x.shape) == 6:
        [batch, Cin, m, _, d, _] = x.shape
        assert x.shape[3] == m
        assert x.shape[-1] == d
        assert Cin % (factor * factor) == 0
        Cout = Cin // (factor * factor)
        assert Cout >= 1
        mout = m * factor
        x = x.contiguous().view(batch, Cout, factor, factor, m, m, d, d)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)  # [batch, Cout, m, factor, m, factor, m, factor, d, d]
        x = x.contiguous().view(batch, Cout, mout, mout, d, d)
    elif len(x.shape) == 5:
        [batch, Cin, m, _, d] = x.shape
        assert x.shape[3] == m
        assert Cin % (factor * factor) == 0
        Cout = Cin // (factor * factor)
        assert Cout >= 1
        mout = m * factor
        x = x.contiguous().view(batch, Cout, factor, factor, m, m, d)
        x = x.permute(0, 1, 4, 2, 5, 3, 6)  # [batch, Cout, m, factor, m, factor, m, factor, d, d]
        x = x.contiguous().view(batch, Cout, mout, mout, d)
    else:
        raise NotImplementedError
    return x


def ltria2so3(L):
    """
    convert from L to SO(3) matrix Q, by:
    1) construct skew symmetric matrix A from L: ltria2skew
    2) cayley map: Q = (I - A) @ (I + A)^(-1)
    :param L: [..., 3]
    :return: SO(3) matrix Q [..., 3, 3]
    """
    B, C, m = assert_ltria(L)
    L = L.contiguous().view(-1, 3)  # [N, d], N = B * C * m * m * m
    # construct the skew-sym matrix
    A = utils.ltria2skew(L)  # [N, 3, 3]
    # cayley map
    Q = utils.CayleyMap(A)  # [N, 3, 3]
    Q = Q.view(B, C, m, m, 3, 3)
    return Q


def so32ltria2(Q):
    """
    convert from SO(3) matrix Q to lower triangle matrix L, by:
    1) cayley map: A = (I - Q) @ (I + Q)^(-1), A is skew-sym matrix
    2) construct lower triangle matrix from skew-sym matrix
    :param SO(3) Q: [..., 3, 3]
    :return: lower triangle matrix L [..., 3]
    """
    B, C, m = assert_spd(Q)
    Q = Q.view(-1, 3, 3)  # [N, 3, 3]
    # cayley map
    A = utils.CayleyMap(Q)  # [N, 3, 3]
    # construct the skew-sym matrix
    L = utils.skew2ltria(A)  # [N, 3]
    L = L.view(B, C, m, m, 3)
    return L


def real2spd(real):
    """
    convert from real matrix to SPD matrix X, by:
    magic code written by Rudra
    :param real: [..., 9]
    :return: SPD matrix X [..., 3, 3]
    """
    # B, C, m = assert_ltria(real)
    B, C, m = assert_real(real, dim=6)
    real = real.contiguous().view(-1, 6)  # [B * C * m * m * m, d], N = B * C * m * m * m
    X = utils.b_real2spd(real)  # [N, 9] -> [N, 3, 3]
    X = X.view(B, C, m, m, 3, 3)
    return X


def spd2real(X):
    """
    convert from SPD matrix X to real matrix real, by:
    magic code written by Rudra
    :param X: shape [..., 3, 3]
    :returns:
        real [..., 9]
    """
    B, C, m = assert_spd(X)
    X = X.contiguous().view(-1, 3, 3)  # [B * C * m * m * m, d, d], N = B * C * m * m * m
    real = utils.b_spd2real(X)  # [N, 9]
    real = real.view(B, C, m, m, 6)
    return real


def assert_spd(X):
    """
    X has very long size, so every time checking is complicate, thus write a function
    :param X: supposed shape [B, C, m, m, 3, 3]
    :returns: B, C, m
    """
    [B, C, m, _, d, _] = X.shape
    assert X.shape[3] == m
    assert X.shape[-1] == d
    assert d == 3
    assert torch.isnan(X).sum().item() == 0
    return B, C, m


def assert_ltria(L):
    """
    :param L: supposed shape [B, C, m, m, 3]
    :return: B, C, m
    """
    [B, C, m, _, d] = L.shape
    assert L.shape[3] == m
    assert d == 3
    return B, C, m


def assert_real(real, dim=3):
    """
    :param real: supposed shape [B, C, m, m, 6]
    :return: B, C, m
    """
    [B, C, m, _, d] = real.shape
    assert real.shape[3] == m
    assert d == dim
    return B, C, m

