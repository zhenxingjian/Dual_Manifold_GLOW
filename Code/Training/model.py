from __future__ import print_function

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import utils
import layers

epsilon = 1e-20
Clp, clp = 0.95, 0.05 # clip to reduce the numerical issues
info = 'DTI2ODF-3D'

param = {
    'eig': 3,
    'dti': 6,
    'odf': 361,
    'reg': 0.001 # the regularization to balance between logz and logdet
}



class m_function(nn.Module):
    def __init__(self, inchannel, outchannel, k_size, unit_factor, nblocks):
        super(m_function, self).__init__()
        self.nblocks = nblocks
        self.layers = []
        self.shortcuts = []
        for b in range(nblocks):
            if b is 0:
                Cin = inchannel
                Cout = int(inchannel * unit_factor)
            else:
                Cin = int(inchannel * unit_factor)
                Cout = int(inchannel * unit_factor)
            self.layers.append(
                nn.Sequential(
                    # weight shape [channel_out, channel_in, kernel_size, kernel_size, kernel_size]
                    nn.Conv2d(Cin, Cout, k_size, padding=(k_size - 1) // 2),  # [Cmid, Cin, k, k, k]
                    nn.BatchNorm2d(Cout),
                    nn.ReLU(),
                    # padding make input feature map size same as ouput feature map size
                    nn.Conv2d(Cout, Cout, 1),  # [Cmid, Cin, k, k, k]
                    nn.BatchNorm2d(Cout)
                )
            )
            self.shortcuts.append(
                nn.Sequential(
                    nn.Conv2d(Cin, Cout, 1),
                    nn.BatchNorm2d(Cout)
                )
            )
        self.layers = nn.ModuleList(self.layers)
        self.shortcuts = nn.ModuleList(self.shortcuts)

        self.final = nn.Conv2d(int(inchannel * unit_factor), outchannel, 3, padding=1)
        nn.init.normal_(self.final.weight, mean=0, std=0.001)
        nn.init.constant_(self.final.bias, 0)

    def block(self, x, block, shortcut):
        xin = x
        for layer in block:
            x = layer(x)
        xin = shortcut(xin)
        x = x + xin
        x = F.relu(x)
        return x

    def forward(self, x):
        """
        :param x: [batch, c, m, n, d]
        :return: the output of convolution
        """
        for b in range(self.nblocks):
            block = self.layers[b]  # which layer block
            shortcut = self.shortcuts[b]  # which shortcut block
            x = self.block(x, block, shortcut)
        x = self.final(x)
        return x


class EIG_AffineCoupling(nn.Module):
    def __init__(self, inchannel, k_size=3, unit_factor=3, nblocks=2):
        super(EIG_AffineCoupling, self).__init__()
        self.inchannel = inchannel
        self.m_function = m_function(inchannel, int(inchannel*2), k_size, unit_factor, nblocks)   # NOTE: hard code for ODF

    def forward(self, x, reverse, logdet):
        B, C, m = layers.assert_real(x, param['eig'])
        d = param['eig']
        x1 = x[:, :C//2, ...]  # [B, C//2, m, m, d]
        x2 = x[:, C//2:, ...]  # [B, C//2, m, m, d]
        x_ = x1.permute(0, 1, 4, 2, 3)  # [B, C//2, d, m, m]
        CC = int(C * d / 2)  # CC = C * d / 2
        x_ = x_.reshape(B, CC, m, m)  # [B, C*d/2, m, m]
        assert CC == self.inchannel
        x_ = self.m_function(x_)  # [B, C*d/2, m, m, m] -> [B, C*d, m, m, m]
        logs = x_[:, :CC, ...]  # [B, C*d/2, m, m, m]
        t = x_[:, CC:, ...]  # [B, C*d/2, m, m, m]
        logs = logs.reshape(B, C//2, d, m, m)
        logs = logs.permute(0, 1, 3, 4, 2)  # [B, C//2, m, m, m, d]
        logs = torch.tanh(logs)
        t = t.reshape(B, C//2, d, m, m)
        t = t.permute(0, 1, 3, 4, 2)  # [B, C//2, m, m, m, d]
        t = torch.tanh(t)

        scale = torch.exp(logs)

        if not reverse:
            x2 =  x2 + t
            x2 = x2 * scale
            x = torch.cat([x1, x2], dim=1)  # -> [B, C, m, m, d]
            dlogdet = logs.sum(dim=(1, 2, 3, 4))  # [B,]
            return x, logdet + param['reg'] * dlogdet
        else:
            x2 = x2 / scale
            x2 = x2 - t
            x = torch.cat([x1, x2], dim=1)  # -> [B, C, m, m, m, d]
            return x


class ODF_AffineCoupling(nn.Module):
    def __init__(self, inchannel, k_size=3, unit_factor=1, nblocks=2):
        super(ODF_AffineCoupling, self).__init__()
        self.inchannel = inchannel
        self.m_function = m_function(inchannel, int(inchannel*2), k_size, unit_factor, nblocks)   # NOTE: hard code for ODF

    def forward(self, x, reverse, logdet):
        B, C, m = layers.assert_real(x, param['odf'])
        d = param['odf']
        x1 = x[:, :C//2, ...]  # [B, C//2, m, m, d]
        x2 = x[:, C//2:, ...]  # [B, C//2, m, m, d]
        x_ = x1.permute(0, 1, 4, 2, 3)  # [B, C//2, d, m, m]
        CC = int(C * d / 2)  # CC = C * d / 2
        x_ = x_.reshape(B, CC, m, m)  # [B, C*d/2, m, m]
        assert CC == self.inchannel
        x_ = self.m_function(x_)  # [B, C*d/2, m, m, m] -> [B, C*d, m, m, m]
        logs = x_[:, :CC, ...]  # [B, C*d/2, m, m, m]
        t = x_[:, CC:, ...]  # [B, C*d/2, m, m, m]
        logs = logs.reshape(B, C//2, d, m, m)
        logs = logs.permute(0, 1, 3, 4, 2)  # [B, C//2, m, m, m, d]
        logs = torch.tanh(logs)
        t = t.reshape(B, C//2, d, m, m)
        t = t.permute(0, 1, 3, 4, 2)  # [B, C//2, m, m, m, d]
        t = torch.tanh(t)

        scale = torch.exp(logs)

        if not reverse:
            x2 =  x2 + t
            x2 = x2 * scale
            x = torch.cat([x1, x2], dim=1)  # -> [B, C, m, m, d]
            dlogdet = logs.sum(dim=(1, 2, 3, 4))  # [B,]
            return x, logdet + dlogdet
        else:
            x2 = x2 / scale
            x2 = x2 - t
            x = torch.cat([x1, x2], dim=1)  # -> [B, C, m, m, m, d]
            return x


class DTI_AffineCoupling(nn.Module):
    def __init__(self, inchannel, k_size=3, unit_factor=6, nblocks=2):
        super(DTI_AffineCoupling, self).__init__()
        self.inchannel = inchannel
        self.m_function = m_function(inchannel, int(inchannel*1.5), k_size, unit_factor, nblocks)  # NOTE: hard code for DTI

    def forward(self, x, reverse, logdet):
        """
        use x2 to calculate the coefficients, and then apply on x1. x2 remain unchanged
        :param x: [B, C, m, m, m, 3, 3]
        :param reverse: bool
        :param logdet: update logdet
        :return: x [B, C, m, m, m, 3, 3]
        """
        B, C, m = layers.assert_spd(x)
        assert int(C * param['dti'] // 2) == self.inchannel, "input channel should be 3 * C, which is 6 * C // 2"
        x1 = x[:, :C // 2, ...]  # [B, C//2, m, m, m, 3, 3]
        x2 = x[:, C // 2:, ...]  # [B, C//2, m, m, m, 3, 3]
        x2_ = layers.spd2real(x2)  # [B, C//2, m, m, m, 6]
        x2_ = x2_.permute(0, 1, 4, 2, 3)  # [B, C//2, 6, m, m, m]
        x2_ = x2_.reshape(B, int(C * param['dti'] // 2), m, m)  # [B, 3*C, m, m, m]
        x_ = self.m_function(x2_)  # [B, 3*C, m, m, m] -> [B, 4.5*C, m, m, m]

        # decompose the output channel into: 1.5C eig, 1.5C L, 4.5C s (which belongs to GL(3))
        real = x_[:, :int(C * param['dti'] // 2), ...]  # [B, 3*C, m, m, m]
        s = x_[:, int(C * param['dti'] // 2):, ...]  # [B, 1.5*C, m, m, m]

        # reshape the three components
        real = real.reshape(B, C//2, param['dti'], m, m)  # [B, C//2, 6, m, m, m]
        real = real.permute(0, 1, 3, 4, 2)  # [B, C//2, m, m, m, 6]
        real = torch.tanh(real)  # NOTE tanh used as constrain the range
        t = layers.real2spd(real)  # [B, C//2, m, m, m, 3, 3]
        s = s.reshape(B, C // 2, 3, m, m)
        s = s.permute(0, 1, 3, 4, 2)  # [B, C//2, m, m, m, 3]
        s = torch.tanh(s)  # NOTE tanh used as constrain the range
        s = layers.ltria2so3(s)  # [B, C//2, m, m, m, 3, 3]

        if not reverse:
            # x1 = x1 + t  # [B, C//2, m, m, m, 3, 3]
            x1 = x1.reshape(-1, 3, 3)  # [N, 3, 3], N = B * C // 2 * m * m * m
            s = s.reshape(-1, 3, 3)  # [N, 3, 3] as well, N = B * C // 2 * m * m
            x1 = s @ x1  # [N, 3, 3]
            x1 = x1 @ s.permute(0, 2, 1)
            x1 = (x1 + x1.permute(0, 2, 1)) / 2  # NOTE: force to be symmetric!
            x1 = x1.reshape(B, C // 2, m, m, 3, 3)
            x1 = x1 + t  # [B, C//2, m, m, m, 3, 3]
            x = torch.cat((x1, x2), dim=1)  # -> [B, C, m, m, m, 3, 3]
            #dlogdet = utils.b_det(s + s.permute(0, 2, 1))# + dlogdet
            #dlogdet = dlogdet.reshape(B, C//2, m, m, m)
            #dlogdet = dlogdet.sum(dim=(1, 2, 3, 4))
            return x, logdet
        else:
            x1 = x1 - t
            s = s.reshape(-1, 3, 3)  # [N, 3, 3]
            s_inv = utils.b_inv33(s)  # [N, 3, 3]
            x1 = x1.reshape(-1, 3, 3)
            x1 = s_inv @ x1  # [N, 3, 3]
            x1 = x1 @ s_inv.permute(0, 2, 1)
            x1 = x1.reshape(B, C // 2, m, m, 3, 3)
            x = torch.cat((x1, x2), dim=1)
            return x


class prior_function(nn.Module):
    def __init__(self, inchannel, outchannel, k_size, unit_factor, nblocks, nsteps):
        """
        :param inchannel: number of channel input
        :param outchannel: number of channel output
        :param k_size: kernel size
        :param unit_factor: channel * unit_factor is the channel in between
        :param nblocks: how many blocks per step
        :param nsteps: how many steps, each step take maxpool once
        """
        super(prior_function, self).__init__()
        self.nblocks = nblocks
        self.nsteps = nsteps
        self.layers = []
        self.shortcuts = []
        for s in range(nsteps):
            for b in range(nblocks):
                if s is 0 and b is 0:
                    Cin = inchannel
                    Cout = int(inchannel * unit_factor)
                else:
                    Cin = int(inchannel * unit_factor)
                    Cout = int(inchannel * unit_factor)
                self.layers.append(
                    nn.Sequential(
                        # weight shape [channel_out, channel_in, kernel_size, kernel_size, kernel_size]
                        nn.Conv2d(Cin, Cout, k_size, padding=(k_size - 1) // 2),  # [Cmid, Cin, k, k, k]
                        nn.BatchNorm2d(Cout),
                        nn.ReLU(),
                        # padding make input feature map size same as ouput feature map size
                        nn.Conv2d(Cout, Cout, 1),  # [Cmid, Cin, k, k, k]
                        nn.BatchNorm2d(Cout)
                    )
                )
                self.shortcuts.append(
                    nn.Sequential(
                        nn.Conv2d(Cin, Cout, 1),
                        nn.BatchNorm2d(Cout)
                    )
                )

        self.layers = nn.ModuleList(self.layers)
        self.shortcuts = nn.ModuleList(self.shortcuts)

        self.final = nn.Conv2d(int(inchannel * unit_factor), outchannel, 1)
        nn.init.normal_(self.final.weight, mean=0, std=0.001)
        nn.init.constant_(self.final.bias, 0)

    def block(self, x, block, shortcut):
        xin = x
        for layer in block:
            x = layer(x)
        xin = shortcut(xin)
        x = x + xin
        x = F.relu(x)
        return x

    def forward(self, x):
        """
        :param x: [batch, c, m, m, d]
        :return: the output of convolution
        """
        where = 0
        for s in range(self.nsteps):
            for b in range(self.nblocks):
                block = self.layers[where]  # which layer block
                shortcut = self.shortcuts[where]  # which shortcut block
                x = self.block(x, block, shortcut)
                where += 1
            #if s != self.nsteps - 1:
            #    x = F.max_pool2d(x, 2)
        x = self.final(x)
        return x


class Glow(nn.Module):
    def __init__(self, num_blocks=2, steps_per_block=2):
        """
        init the glow model
        :param num_blocks: number of blocks, default 3
        :param steps_per_block: in each block, how many step there is
        """
        super(Glow, self).__init__()
        self.num_blocks = num_blocks
        self.steps_per_block = steps_per_block

        # self.dim_reduce = DimReduction(param['dim'])

        def init_glow(num_blocks, steps_per_block, name):
            layers = []
            for b in range(num_blocks):
                channel = 2 ** (b + 2)  # s=0, c=8; s=1, c=32; s=2, c=128
                for s in range(steps_per_block):
                    if name is 'dti':
                        layers.append(
                            nn.Sequential(  # there are num_blocks * steps_per_block sequential block
                                DTI_AffineCoupling(inchannel=int(channel*param['dti']//2))
                            )
                        )
                    elif name is 'odf':
                        layers.append(
                            nn.Sequential(  # there are num_blocks * steps_per_block sequential block
                                ODF_AffineCoupling(inchannel=channel*param['odf']//2)
                            )
                        )
                    elif name is 'eig':
                        layers.append(
                            nn.Sequential(  # there are num_blocks * steps_per_block sequential block
                                EIG_AffineCoupling(inchannel=channel*param['eig']//2)
                            )
                        )
                    else:
                        raise NotImplementedError
            return layers

        self.dti2odf = prior_function(
            inchannel=param['dti']+param['eig'], outchannel=2*param['odf'], k_size=3, unit_factor=12, nblocks=1, nsteps=3)

        self.odflayers = init_glow(num_blocks, steps_per_block, 'odf')
        self.odflayers = nn.ModuleList(self.odflayers)

        self.dtilayers = init_glow(num_blocks, steps_per_block, 'dti')
        self.dtilayers = nn.ModuleList(self.dtilayers)

        self.eiglayers = init_glow(num_blocks, steps_per_block, 'eig')
        self.eiglayers = nn.ModuleList(self.eiglayers)

    def logmap(self, odf):  # NOTE
        """
        :param odf: [B, C, m, m, m, d]
        """
        d = param['odf'] + 1
        B, C, m = layers.assert_real(odf, d)
        north_pole = torch.zeros(1, 1, 1, 1, d).to(odf.device)
        north_pole[..., 0] = 1
        theta = torch.acos(odf[..., 0])  # [B, C, m, m, m]
        cos_theta = odf[..., 0]  # [B, C, m, m, m]
        sin_theta = torch.sin(theta)  # [B, C, m, m, m]
        odf = odf - north_pole * cos_theta.unsqueeze(-1)
        odf = odf * theta.unsqueeze(-1) / (sin_theta.unsqueeze(-1) + epsilon)
        odf = odf[..., 1:]  # [B, C, m, m, m, d-1]
        return odf

    def expmap(self, v):
        B, C, m = layers.assert_real(v, param['odf'])
        d = param['odf'] + 1
        north_pole = torch.zeros(1, 1, 1, 1, d).to(v.device)
        north_pole[..., 0] = 1
        v_add_dim = torch.zeros(B, C, m, m, d).to(v.device)
        v_add_dim[..., 1:] = v
        odf = torch.zeros(B, C, m, m, d).to(v.device)
        v_norm = torch.norm(v_add_dim, dim=-1).unsqueeze(-1)  # [B, C, m, m, m]
        cos_norm = torch.cos(v_norm)
        sin_norm = torch.sin(v_norm)
        odf = cos_norm * north_pole + sin_norm * v_add_dim / (v_norm + epsilon)
        odf[..., 0] = torch.abs(odf[..., 0])
        return odf

    def split(self, x, z):
        """
        :param x: [B, C, m, m, ...]
        :param z: [B, C', m, m, ...]
        :return:
        """
        C = x.shape[1]
        x_1 = x[:, :C // 2, ...]
        x_2 = x[:, C // 2:, ...]
        x = layers.squeeze2d(x_1)  # [B, C//2, m, m, m, ...] -> [B, 2 * C, m // 2, m // 2, m // 2, ...]
        if z is None:
            z = x_2
        else:
            z = torch.cat([x_2, z], dim=1)  # [B, C' + C // 2, m, m, m, ...]
        z = layers.squeeze2d(z)
        return x, z

    def split_reverse(self, x, z):
        x_1 = layers.unsqueeze2d(x)
        C = x_1.shape[1]
        z = layers.unsqueeze2d(z)
        x_2 = z[:, :C, ...]
        x = torch.cat([x_1, x_2], dim=1)
        z = z[:, C:, ...]  # may end up being [B, 0, ...]
        return x, z

    def block_forward(self, x, block_lst, logdet):
        assert len(block_lst) == 1
        # [an_method, invert_method, coupling_method] = block_lst
        [coupling_method] = block_lst
        x, logdet = coupling_method(x, False, logdet)
        return x, logdet

    def block_reverse(self, x, block_lst, logdet):
        # [an_method, invert_method, coupling_method] = block_lst
        [coupling_method] = block_lst
        x = coupling_method(x, True, logdet)
        return x

    def prior(self, dti, eig, odf=None, sample=False):
        """
        only define distribution on y, shape [B, C, m, m, 2],
        use convolution to learn the distribution and conditioned on x's distribution
        """
        B, C, m = layers.assert_real(dti, param['dti'])
        gdti = layers.gaussian_real(
            mean=torch.zeros_like(dti).to(dti.device), 
            logsd=torch.zeros_like(dti).to(dti.device),
            dim=param['dti']
        )

        geig = layers.gaussian_real(
            mean=torch.zeros_like(eig).to(eig.device),
            logsd=torch.zeros_like(eig).to(eig.device),
            dim=param['eig']
        )

        y = torch.cat([dti, eig], dim=-1)  # [B, C, m, m, dti+eig]
        y = y.reshape(B * C, m, m, param['dti']+param['eig'])
        y = y.permute(0, 3, 1, 2)  # [B * C, dti+eig, m, m]
        y = self.dti2odf(y)  # [B * C, 2*odf, m, m]
        y = y.reshape(B, C, 2 * param['odf'], m, m)
        y = y.permute(0, 1, 3, 4, 2)  # [B, C, m, m, 2 * odf]
        xmean = y[..., :param['odf']]  # [B, C, m, m, odf]
        xlogstd = y[..., param['odf']:]
        # xlogstd = torch.tanh(xlogstd)
        godf = layers.gaussian_real(
            mean = xmean,
            logsd = xlogstd,
            dim=param['odf']
        )
        if not sample:
            logpdti = gdti.logp(dti)
            logpeig = geig.logp(eig)
            logpodf = godf.logp(odf)
            return param['reg'] * logpdti + param['reg'] * logpeig + logpodf
        else:
            sample = godf.sample
            return sample


    def forward(self, odf, dti, eig, extra = None, reverse = False, reconstruct = False):
        """
        :param dti: [B, C, m, m, 3, 3], original data
        :param odf: [B, C, m, m, d], observed data
        :param eig: [B, C, m, m, 3]
        :return:
        """

        if reverse:
            return self.reverse(odf, dti, eig, extra)

        if reconstruct:
            return self.reconstruct(dti = odf, eig = dti) # for multiGPU, the first two input is actually the dti, eig

        odf = odf.permute(0,4,1,2,3,5)
        dti = dti.permute(0,4,1,2,3,5,6)
        eig = eig.permute(0,4,1,2,3,5)

        odf = odf.reshape([-1, 1, 24, 24, 362])
        dti = dti.reshape([-1, 1, 24, 24, 3,3])
        eig = eig.reshape([-1, 1, 24, 24, 3])


        B = odf.shape[0]
        logdet = torch.zeros(B).to(odf.device)
        odfz, dtiz, eigz = None, None, None

        odf = self.logmap(odf)# torch.log(odf + epsilon)
        eig = torch.log(eig + epsilon)
        odf = layers.squeeze2d(odf)  # [B, 1, 32, 32, 3, 3] -> [B, 4, 16, 16, 3, 3]
        dti = layers.squeeze2d(dti)  # [B, 1, 32, 32, 362] -> [B, 4, 16, 16, 362]
        eig = layers.squeeze2d(eig)  # [B, 1, 32, 32, 3] -> [B, 4, 16, 16, 3]

        where = 0
        for b in range(self.num_blocks):
            for s in range(self.steps_per_block):
                odf, logdet = self.block_forward(odf, self.odflayers[where], logdet)
                dti, logdet = self.block_forward(dti, self.dtilayers[where], logdet)
                eig, logdet = self.block_forward(eig, self.eiglayers[where], logdet)
                where += 1
            if b is not self.num_blocks - 1:
                odf, odfz = self.split(odf, odfz)
                dti, dtiz = self.split(dti, dtiz)
                eig, eigz = self.split(eig, eigz)

        C = odf.shape[1]
        odff = torch.cat([odf, odfz], dim=1)
        dtii = torch.cat([dti, dtiz], dim=1)
        dtii = layers.spd2real(dtii)
        eigg = torch.cat([eig, eigz], dim=1)

        logpz = self.prior(dtii, eigg, odff, False)
        dtii = layers.real2spd(dtii)
        return logdet, logpz, odff, dtii, eigg

    def reverse(self, odff, dtii, eigg, C):
        """
        :param xz: the hidden tensor for x
        :param yz: the hidden tensor for y
        :param x: original signal
        :param y: compressed signal
        :return: reconstruction error
        """
        dti = dtii[:, :C, ...]
        dtiz = dtii[:, C:, ...]
        odf = odff[:, :C, ...]
        odfz = odff[:, C:, ...]
        eig = eigg[:, :C, ...]
        eigz = eigg[:, C:, ...]

        B = dtii.shape[0]
        logdet = torch.zeros(B).to(odff.device)

        where = len(self.odflayers) - 1
        for b in range(self.num_blocks):
            for s in range(self.steps_per_block):
                odf = self.block_reverse(odf, self.odflayers[where], logdet)
                dti = self.block_reverse(dti, self.dtilayers[where], logdet)
                eig = self.block_reverse(eig, self.eiglayers[where], logdet)
                where -= 1
            if b is not self.num_blocks - 1:
                odf, odfz = self.split_reverse(odf, odfz)
                dti, dtiz = self.split_reverse(dti, dtiz)
                eig, eigz = self.split_reverse(eig, eigz)

        odf = layers.unsqueeze2d(odf)  # [B, 1, 32, 32] -> [B, 4, 16, 16]
        dti = layers.unsqueeze2d(dti)
        eig = layers.unsqueeze2d(eig)
        odf = self.expmap(odf) #torch.exp(odf)
        eig = torch.exp(eig)

        odf = odf.reshape([-1, 24, 1, 24, 24, 362])
        dti = dti.reshape([-1, 24, 1, 24, 24, 3,3])
        eig = eig.reshape([-1, 24, 1, 24, 24, 3])

        odf = odf.permute(0,2,3,4,1,5)
        dti = dti.permute(0,2,3,4,1,5,6)
        eig = eig.permute(0,2,3,4,1,5)


        return odf, dti, eig

    def reconstruct(self, dti, eig):
        """
        give observed y, reconstruct x
        :return: reconstruction error
        """
        dti = dti.permute(0,4,1,2,3,5,6)
        eig = eig.permute(0,4,1,2,3,5)

        dti = dti.reshape([-1, 1, 24, 24, 3,3])
        eig = eig.reshape([-1, 1, 24, 24, 3])


        B, C, m = layers.assert_spd(dti)
        logdet = torch.zeros(B).to(dti.device)

        # y passing forward
        dti = layers.squeeze2d(dti)
        eig = torch.log(eig + epsilon)
        eig = layers.squeeze2d(eig)

        dtiz, eigz = None, None
        where = 0
        for b in range(self.num_blocks):
            for s in range(self.steps_per_block):
                dti, logdet = self.block_forward(dti, self.dtilayers[where], logdet)
                eig, logdet = self.block_forward(eig, self.eiglayers[where], logdet)
                where += 1
            if b is not self.num_blocks - 1:
                dti, dtiz = self.split(dti, dtiz)
                eig, eigz = self.split(eig, eigz)

        C = dti.shape[1]
        dtii = torch.cat([dti, dtiz], dim=1)
        dtii = layers.spd2real(dtii)
        eigg = torch.cat([eig, eigz], dim=1)
        odff = self.prior(dtii, eigg, None, True)
        odf = odff[:, :C, ...]
        odfz = odff[:, C:, ...]

        # x passing backward
        where = len(self.odflayers) - 1
        for b in range(self.num_blocks):
            for s in range(self.steps_per_block):
                odf = self.block_reverse(odf, self.odflayers[where], logdet)
                where -= 1
            if b is not self.num_blocks - 1:
                odf, odfz = self.split_reverse(odf, odfz)

        odf = layers.unsqueeze2d(odf)  # [B, 1, 32, 32] -> [B, 4, 16, 16]
        odf = self.expmap(odf)#torch.exp(odf)

        odf = odf.reshape([-1, 24, 1, 24, 24, 362])
        odf = odf.permute(0,2,3,4,1,5)
        
        return odf
