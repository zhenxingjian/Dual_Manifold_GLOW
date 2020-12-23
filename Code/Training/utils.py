# In this script, there shouldn't be any function specify size
from __future__ import division

import torch
import numpy as np
import torch.utils.data as tud
import torch.nn.functional as F

Eps = 1e-6
eps = 1e-20


class Dataset(tud.Dataset):
    def __init__(self, X, Y):
        super(Dataset, self).__init__()
        self.X = torch.from_numpy(X).float()  # [1000, 32, 32, 32, 3, 3]
        self.Y = torch.from_numpy(Y).float()  # [1000, 32, 32, 32, 3, 3]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        x = self.X[item]  # [32, 32, 32, 3, 3]
        y = self.Y[item]  # [32, 32, 32, 3, 3]
        return x, y


class MOREDataset(tud.Dataset):
    def __init__(self, ODF, DTI, EIG):
        super(MOREDataset, self).__init__()
        self.ODF = torch.from_numpy(ODF).float()
        self.DTI = torch.from_numpy(DTI).float()
        self.EIG = torch.from_numpy(EIG).float()

    def __len__(self):
        return self.ODF.shape[0]

    def __getitem__(self, item):
        odf = self.ODF[item]
        dti = self.DTI[item]
        eig = self.EIG[item]
        return odf, dti, eig


class REFDataset(tud.Dataset):
    def __init__(self, ODF, DTI, ODFREF):
        super(REFDataset, self).__init__()
        self.ODF = torch.from_numpy(ODF).float()
        self.DTI = torch.from_numpy(DTI).float()
        self.ODFREF = torch.from_numpy(ODFREF).float()

    def __len__(self):
        return self.ODF.shape[0]

    def __getitem__(self, item):
        odf = self.ODF[item]
        dti = self.DTI[item]
        odfref = self.ODFREF[item]
        return odf, dti, odfref


def eig2Eig(eig):
    """
    assume eig has already passed the assertion check
    :param eig: diagonal eigenvalue matrix [N, 3]
    :return: Eig [N, 3, 3]
    """
    assert len(eig.shape) == 2
    N = eig.shape[0]
    Eig = torch.zeros(N, 3, 3).to(eig.device)  # [N, 3, 3]
    Eig[:, 0, 0] = eig[:, 0]
    Eig[:, 1, 1] = eig[:, 1]
    Eig[:, 2, 2] = eig[:, 2]
    return Eig


def ltria2skew(L):
    """
    assume L has already passed the assertion check
    :param L: lower triangle matrix, shape [N, 3]
    :return: skew sym A [N, 3, 3]
    """
    if len(L.shape) == 2:
        N = L.shape[0]
        # construct the skew-sym matrix
        A = torch.zeros(N, 3, 3).cuda()  # [N, 3, 3]
        A[:, 1, 0] = L[:, 0]
        A[:, 2, 0] = L[:, 1]
        A[:, 2, 1] = L[:, 2]
        A[:, 0, 1] = -L[:, 0]
        A[:, 0, 2] = -L[:, 1]
        A[:, 1, 2] = -L[:, 2]
        return A
    elif len(L.shape) == 1:
        A = torch.zeros(3, 3).cuda()
        A[1, 0] = L[0]
        A[2, 0] = L[1]
        A[2, 1] = L[2]
        A[0, 1] = -L[0]
        A[0, 2] = -L[1]
        A[1, 2] = -L[2]
        return A
    else:
        raise NotImplementedError


def skew2ltria(A):
    """
    assume A has already passed the assertion check
    :param A: skew sym matrix, shape [N, 3, 3]
    :return: lower triangle matrix A [N, 3]
    """
    assert len(A.shape) == 3
    N = A.shape[0]
    # construct the lower triangle matrix
    L = torch.zeros(N, 3).cuda()  # [N, 3]
    L[:, 0] = A[:, 1, 0]
    L[:, 1] = A[:, 2, 0]
    L[:, 2] = A[:, 2, 1]
    return L


def CayleyMap(B):
    """
    Cayley Map(B) = (I-B) * (I+B)^(-1) in SO(n), or from SO(n) to skew-symmetric matrix
    :param B: skew symmetric matrix [N, n, n], where -B = B^T, or SO(n) matrix, where B.T @ B = I
    :return:
        SO(n) matrix [N, n, n] or skew-sym matrix [N, n, n]
    """
    n = B.shape[-1]
    two = False
    if len(B.shape) == 2:
        B = B[None, :, :]
        two = True
    I = torch.eye(n).float().unsqueeze(0).to(B.device)  # [1, n, n]
    left = I - B
    if n == 3:
        right = b_inv33(I + B)
    elif n == 2:
        right = b_inv(I + B)
    else:
        right = torch.inverse(I + B)
    if not two:
        return torch.bmm(left, right)  # [N, n, n]
    else:
        return torch.bmm(left, right).squeeze(0)


def b_inv(b_mat):
    """
    inverse function for 2x2 matrix
    :param b_mat: [M, 2, 2]
    :return: [M, 2, 2]
    """
    b00 = b_mat[:, 0, 0]
    b01 = b_mat[:, 0, 1]
    b10 = b_mat[:, 1, 0]
    b11 = b_mat[:, 1, 1]
    det = (b00 * b11 - b01 * b10)
    b00 = b00 / (det + eps)
    b01 = b01 / (det + eps)
    b10 = b10 / (det + eps)
    b11 = b11 / (det + eps)
    b_inv1 = torch.cat((torch.cat((b11.view(-1, 1, 1), -1. * b01.view(-1, 1, 1)), dim=2),
                        torch.cat((-1. * b10.view(-1, 1, 1), b00.view(-1, 1, 1)), dim=2)), dim=1)
    return b_inv1


def b_inv33(b_mat):
    """
    inverse function for 3x3 matrix
    :param b_mat: [M, 3, 3]
    :return: [M, 3, 3]
    """
    assert len(b_mat.shape) == 3
    assert b_mat.shape[1] == 3
    assert b_mat.shape[2] == 3
    b00 = b_mat[:, 0, 0]
    b01 = b_mat[:, 0, 1]
    b02 = b_mat[:, 0, 2]
    b10 = b_mat[:, 1, 0]
    b11 = b_mat[:, 1, 1]
    b12 = b_mat[:, 1, 2]
    b20 = b_mat[:, 2, 0]
    b21 = b_mat[:, 2, 1]
    b22 = b_mat[:, 2, 2]
    det = (b00 * (b11 * b22 - b12 * b21) - b01 * (b10 * b22 - b12 * b20) + b02 * (b10 * b21 - b11 * b20))
    c00 = b11 * b22 - b12 * b21
    c01 = b02 * b21 - b01 * b22
    c02 = b01 * b12 - b02 * b11
    c10 = b12 * b20 - b10 * b22
    c11 = b00 * b22 - b02 * b20
    c12 = b02 * b10 - b00 * b12
    c20 = b10 * b21 - b11 * b20
    c21 = b01 * b20 - b00 * b21
    c22 = b00 * b11 - b01 * b10
    c00 = (c00 / (det + eps)).view(-1, 1, 1)
    c01 = (c01 / (det + eps)).view(-1, 1, 1)
    c02 = (c02 / (det + eps)).view(-1, 1, 1)
    c10 = (c10 / (det + eps)).view(-1, 1, 1)
    c11 = (c11 / (det + eps)).view(-1, 1, 1)
    c12 = (c12 / (det + eps)).view(-1, 1, 1)
    c20 = (c20 / (det + eps)).view(-1, 1, 1)
    c21 = (c21 / (det + eps)).view(-1, 1, 1)
    c22 = (c22 / (det + eps)).view(-1, 1, 1)
    b_inv1 = torch.cat(
        (torch.cat((c00, c01, c02), dim=2), torch.cat((c10, c11, c12), dim=2), torch.cat((c20, c21, c22), dim=2)),
        dim=1)
    return b_inv1


def b_det(b_mat):
    """
    :param b_mat: shape [B, 3, 3]
    :return: [B]
    """
    assert len(b_mat.shape) == 3
    assert b_mat.shape[1] == 3
    assert b_mat.shape[2] == 3
    b00 = b_mat[:, 0, 0]
    b01 = b_mat[:, 0, 1]
    b02 = b_mat[:, 0, 2]
    b10 = b_mat[:, 1, 0]
    b11 = b_mat[:, 1, 1]
    b12 = b_mat[:, 1, 2]
    b20 = b_mat[:, 2, 0]
    b21 = b_mat[:, 2, 1]
    b22 = b_mat[:, 2, 2]
    bdet = b00 * b11 * b22 + b10 * b21 * b02 + b20 * b01 * b12
    bdet = bdet - b02 * b11 * b20 - b01 * b10 * b22 - b00 * b21 * b12
    return bdet


def get_eig(X):
    """
    :param X: shape [N, 2, 2]
    :return:
    """
    assert X.shape[1] == X.shape[2]
    assert X.shape[1] == 2
    a, b, c = X[:, 0, 0], X[:, 0, 1], X[:, 1, 1]  # each shape [N,]
    assert (X[:, 1, 0] - b).max() < eps  # need to be symmetric
    delta = (a - c) ** 2 + 4 * b ** 2
    delta = F.relu(delta)  # for numerical issue
    eig1 = ((a + c) + torch.sqrt(delta)) / 2
    eig2 = ((a + c) - torch.sqrt(delta)) / 2
    return eig1.view(-1, 1), eig2.view(-1, 1)


def get_vec(X, eig):
    """
    :param X: [N, 2, 2]
    :param eig: [N,]
    :return:
        vec: cos expression of X
    """
    assert X.shape[1] == X.shape[2]
    assert X.shape[1] == 2
    N = X.shape[0]
    a, b = X[:, 0, 0], X[:, 0, 1]  # each shape [N,]
    assert (X[:, 1, 0] - b).max() < eps  # need to be symmetric
    assert eig.shape[0] == N
    where = (b != 0) or (a != eig)
    vec = torch.ones(N, 3).to(X.device)
    sin = eig[where] - a[where]
    cos = b[where]
    vec[where] = cos / torch.sqrt(sin ** 2 + cos ** 2)
    return vec


def b_spd2real(X):
    """
    use LU decomposition
    [a1*a1  a1*b1       a1*b2                   [a1           [a1 b1  b2
     a1*b1  b1*b1+a2*a2 b1*b2+a2*b3         =    b1 a2      @     a2  b3
     a1*b2  b1*b2+a2*b3 b2*b2+b3*b3+a3*a3]       b2 b3  a3]           a3]
    :param X: [N, 3, 3], spd matrix
    :return: Y [N, 6], which is the reparametrization of X [√a1, √a2, √a3, b1, b2, b3]
    """
    X = (X + X.permute(0, 2, 1)) / 2
    assert X.shape[1] == X.shape[2]
    assert X.shape[1] == 3
    assert torch.isnan(X).sum().item() == 0
    assert (X - X.permute(0, 2, 1)).max() <= Eps  # check symmetric
    a_1 = torch.sqrt(F.relu(X[:, 0, 0]) + eps ) 
    b_1 = X[:, 0, 1] / (a_1 + eps)
    b_2 = X[:, 0, 2] / (a_1 + eps)
    a_2 = torch.sqrt(F.relu(X[:, 1, 1] - b_1 * b_1) + eps)
    b_3 = (X[:, 1, 2] - b_1 * b_2) / (a_2 + eps)
    a_3 = torch.sqrt(F.relu(X[:, 2, 2] - b_2 * b_2 - b_3 * b_3) + eps )
    real = [torch.sqrt(a_1), torch.sqrt(a_2), torch.sqrt(a_3), b_1, b_2, b_3]
    real = torch.stack(real, dim=1)  # [N, 6]
    assert torch.isnan(real).sum().item() == 0
    return real


def b_real2spd(X):
    """
    :param X: [N, 6]
    :return: Y [N, 3, 3], which is the SPD
    """
    assert torch.isnan(X).sum().item() == 0
    N = X.shape[0]
    # give me a_1, a_2, a_3, b_1, b_2, b_3
    a_1, a_2, a_3 = X[:, 0] ** 2, X[:, 1] ** 2, X[:, 2] ** 2
    b_1, b_2, b_3 = X[:, 3], X[:, 4], X[:, 5]
    # construct LU
    L = torch.zeros(N, 3, 3).to(X.device)
    L[:, 0, 0], L[:, 1, 1], L[:, 2, 2] = a_1, a_2, a_3
    L[:, 1, 0], L[:, 2, 0], L[:, 2, 1] = b_1, b_2, b_3
    Y = L @ L.permute(0, 2, 1)  # [N, 3, 3] @ [N, 3, 3] -> [N, 3, 3]
    assert torch.isnan(Y).sum().item() == 0
    return Y


def spdist(X, Y):
    """
    function to measure distance square on SPD
    :param X: [N, 3, 3]
    :param Y: [N, 3, 3]
    :return: [N]
    """
    N = X.shape[0]
    assert X.shape == Y.shape
    Z = X - Y
    Z = Z ** 2  # [N, 3, 3]
    Z = Z.sum(dim=(1, 2))  # [N]
    return Z
