# python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/10 15:45
# @Author  : Liupeng Lin
# @Email   : linliupeng@whu.edu.cn
# Copyright (C) 2021 Liupeng Lin. All Rights Reserved.


import torch
from torch import nn
import torch.nn.init as init


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        return self.conv1(x)


class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out = torch.add(x, self.conv2(self.conv1(x)))
        return out


class SALayer(nn.Module):
    def __init__(self, nFeat):
        super(SALayer, self).__init__()
        self.sal_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.sal_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=3, padding=1, bias=True), nn.Sigmoid())
        self.sal_conv3 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        out1 = self.sal_conv1(x)
        sal_weight = self.sal_conv2(out1)
        out = self.sal_conv3(torch.mul(x, sal_weight))
        return out


class ComplexBlock(nn.Module):
    def __init__(self, nFeat):
        super(ComplexBlock, self).__init__()
        nFeat_ = nFeat//2
        self.convs1 = nn.Sequential(
            nn.Conv2d(1, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs2 = nn.Sequential(
            nn.Conv2d(2, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs3 = nn.Sequential(
            nn.Conv2d(2, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs4 = nn.Sequential(
            nn.Conv2d(1, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs5 = nn.Sequential(
            nn.Conv2d(2, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.convs6 = nn.Sequential(
            nn.Conv2d(1, nFeat_, kernel_size=3, padding=1, bias=True), nn.PReLU())
        self.conv_cb = nn.Sequential(
            nn.Conv2d(nFeat_*6, nFeat, kernel_size=3, padding=1, bias=True), nn.PReLU())

    def forward(self, x):
        conv1 = self.convs1(torch.index_select(x, 1, torch.LongTensor([0]).cuda()))
        conv2 = self.convs2(torch.index_select(x, 1, torch.LongTensor([1, 2]).cuda()))
        conv3 = self.convs3(torch.index_select(x, 1, torch.LongTensor([3, 4]).cuda()))
        conv4 = self.convs4(torch.index_select(x, 1, torch.LongTensor([5]).cuda()))
        conv5 = self.convs5(torch.index_select(x, 1, torch.LongTensor([6, 7]).cuda()))
        conv6 = self.convs6(torch.index_select(x, 1, torch.LongTensor([8]).cuda()))
        out = self.conv_cb(torch.cat((conv1, conv2, conv3, conv4, conv5, conv6), 1))
        return out


class LPSR(nn.Module):
    def __init__(self, nFeat):
        super(LPSR, self).__init__()
        self.lpsr_cb = ComplexBlock(nFeat)
        self.lpsr_deconv = nn.Sequential(
            nn.ConvTranspose2d(nFeat, nFeat, kernel_size=4, stride=2, padding=1, bias=True), nn.PReLU())
        self.lpsr_res = self.make_layer(ResBlock, 5)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.lpsr_deconv(self.lpsr_cb(x))
        out2 = self.lpsr_res(out1)
        out = torch.add(out1, out2)
        return out


class HSFE(nn.Module):
    def __init__(self, args):
        super(HSFE, self).__init__()
        nFeat = args.nFeat
        ncha_sinsar = args.ncha_sinsar
        self.hsfe_conv1 = nn.Sequential(
            nn.Conv2d(ncha_sinsar, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())
        self.hsfe_res = self.make_layer(ResBlock, 5)
        self.hsfe_sal = SALayer(nFeat)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.hsfe_conv1(x)
        out2 = self.hsfe_sal(self.hsfe_res(out1))
        out = torch.add(out1, out2)
        return out


class HSSA(nn.Module):
    def __init__(self, args):
        super(HSSA, self).__init__()
        nFeat = args.nFeat
        ncha_sinsar = args.ncha_sinsar
        ncha_polsar = args.ncha_polsar

        self.hssa_deconv = nn.Sequential(
            nn.ConvTranspose2d(nFeat, nFeat, kernel_size=4, stride=2, padding=1, bias=True), nn.PReLU())
        self.hssa_conv1 = nn.Sequential(
            nn.Conv2d(ncha_sinsar, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())
        self.hssa_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, 1, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU(), nn.Sigmoid())
        self.hssa_conv3 = nn.Sequential(
            nn.Conv2d(ncha_polsar, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())
        self.hssa_conv4 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, lr, hr):
        out1 = self.hssa_conv2(self.hssa_conv1(hr))
        out2 = self.hssa_deconv(self.hssa_conv3(lr))
        out = self.hssa_conv4(torch.mul(out1, out2))
        return out


class LPCA(nn.Module):
    def __init__(self, args):
        super(LPCA, self).__init__()
        nFeat = args.nFeat
        ncha_sinsar = args.ncha_sinsar
        ncha_polsar = args.ncha_polsar
        
        self.lpca_deconv = nn.Sequential(
            nn.ConvTranspose2d(nFeat, nFeat, kernel_size=4, stride=2, padding=1, bias=True), nn.PReLU(), nn.Sigmoid())
        self.lpca_conv1 = nn.Sequential(
            nn.Conv2d(ncha_polsar, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.lpca_conv2 = nn.Sequential(
            nn.Conv2d(ncha_sinsar, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())
        self.lpca_conv3 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3,  stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, lr, hr):
        avg_weight = self.lpca_deconv(self.lpca_conv1(lr))
        out1 = self.lpca_conv2(hr)
        out = self.lpca_conv3(torch.mul(out1, avg_weight))
        return out


class CroAM(nn.Module):
    def __init__(self, args):
        super(CroAM, self).__init__()
        nFeat = args.nFeat
        self.croam_hssa = HSSA(args)
        self.croam_lpca = LPCA(args)
        self.croam_conv1 = nn.Sequential(
            nn.Conv2d(nFeat*2, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())

    def forward(self, lr, hr):
        out = self.croam_conv1(torch.cat((self.croam_hssa(lr, hr), self.croam_lpca(lr, hr)), 1))
        return out


class PSFN(nn.Module):
    def __init__(self, args):
        super(PSFN, self).__init__()
        nFeat = args.nFeat
        ncha_polsar = args.ncha_polsar
        self.psfn_hsfe = HSFE(args)
        self.psfn_lpsr = LPSR(nFeat)
        self.psfn_croam = CroAM(args)
        self.psfn_conv1 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())
        self.psfn_conv2 = nn.Sequential(
            nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())
        self.psfn_conv3 = nn.Sequential(
            nn.Conv2d(nFeat, ncha_polsar, kernel_size=3, stride=1, padding=1, bias=True), nn.PReLU())
        self.psfn_res1 = self.make_layer(ResBlock, 3)
        self.psfn_res2 = self.make_layer(ResBlock, 3)
        self._initialize_weights()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, lr, hr):
        out1 = self.psfn_res1(self.psfn_conv1(torch.add(self.psfn_lpsr(lr), self.psfn_hsfe(hr))))
        out2 = self.psfn_croam(lr, hr)
        out = self.psfn_conv3(self.psfn_conv2(self.psfn_res2(torch.add(out1, out2))))
        return out
