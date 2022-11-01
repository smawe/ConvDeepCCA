# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:20:08 2022

@author: mawes
"""

import torch
import torch.nn as nn
from objectives import cca_loss

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class DownsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=False):
        super(DownsamplingUnit, self).__init__()

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        model = [nn.Conv2d(channels_in, channels_in, kernel_size=3, stride=1,
                           padding=1,
                           dilation=1,
                           groups=channels_in if groups else 1,
                           bias=bias,
                           padding_mode='reflect')]

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_in, affine=True))

        model.append(nn.LeakyReLU(inplace=False))
        model.append(nn.Conv2d(channels_in, channels_out, kernel_size=3,
                               stride=1,
                               padding=1,
                               dilation=1,
                               groups=channels_in if groups else 1,
                               bias=bias,
                               padding_mode='reflect'))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        fx = self.downsample(x)
        fx = self.model(fx)
        return fx

    

class UpsamplingUnit(nn.Module):
    def __init__(self, channels_in, channels_out, groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=True):
        super(UpsamplingUnit, self).__init__()

        model = [nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1,
                           padding=1,
                           dilation=1,
                           groups=channels_in if groups else 1,
                           bias=bias,
                           padding_mode='reflect')]

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        model.append(nn.Conv2d(channels_out, channels_out, kernel_size=3,
                               stride=1,
                               padding=1,
                               dilation=1,
                               groups=channels_in if groups else 1,
                               bias=bias,
                               padding_mode='reflect'))

        if batch_norm:
            model.append(nn.BatchNorm2d(channels_out, affine=True))

        model.append(nn.LeakyReLU(inplace=False))

        if dropout > 0.0:
            model.append(nn.Dropout2d(dropout))

        self.model = nn.Sequential(*model)
        self.upsample = nn.ConvTranspose2d(channels_out, channels_out,
                                           kernel_size=2,
                                           stride=2,
                                           padding=0,
                                           dilation=1,
                                           groups=channels_in if groups else 1,
                                           bias=bias)

    def forward(self, x):
        fx = self.model(x)
        fx = self.upsample(fx)
        return fx

class Analyzer(nn.Module):
    def __init__(self, channels_org=3, channels_net=8, depth=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 bias=False,
                 **kwargs):
        super(Analyzer, self).__init__()

        # Initial color convertion
        self.embedding = nn.Sequential(
            nn.Conv2d(channels_org, channels_net, kernel_size=3, stride=1,
                      padding=1,
                      dilation=1,
                      groups=channels_org if groups else 1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, channels_net, kernel_size=3, stride=1,
                      padding=1,
                      dilation=1,
                      groups=channels_org if groups else 1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, channels_net, kernel_size=3, stride=1,
                      padding=1,
                      dilation=1,
                      groups=channels_org if groups else 1,
                      bias=bias,
                      padding_mode='reflect'))

        down_track = [DownsamplingUnit(
            channels_in=channels_net * channels_expansion ** i,
            channels_out=channels_net * channels_expansion ** (i+1), 
            groups=groups,
            batch_norm=batch_norm,
            dropout=dropout,
            bias=bias)
                      for i in range(depth)]

        # Final convolution in the analysis track
        self.analysis_track = nn.ModuleList(down_track)

        self.apply(initialize_weights)

    def forward(self, x):
        fx = self.embedding(x)

        # Store the output of each layer as bridge connection to the synthesis
        # track
        fx_brg_list = []
        for i, layer in enumerate(self.analysis_track):
            fx_brg_list.append(fx)
            fx = layer(fx)

        return fx, fx_brg_list
    
class Synthesizer(nn.Module):
    def __init__(self, classes=1, channels_net=8,
                 depth=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 dropout=0.0,
                 autoencoder_channels_net=None,
                 autoencoder_channels_expansion=None,
                 use_bridge=True,
                 trainable_bridge=False,
                 bias=False,
                 **kwargs):
        super(Synthesizer, self).__init__()


        if use_bridge:
            input_channels_mult = 2
        else:
            input_channels_mult = 1

        self.embedding = nn.ConvTranspose2d(
            channels_net * channels_expansion ** depth,
            channels_net
            * channels_expansion**(depth-1),
            kernel_size=2,
            stride=2,
            padding=0,
            groups=channels_net * channels_expansion ** depth if groups else 1,
            bias=bias)

        # Initial deconvolution in the synthesis track
        up_track = [UpsamplingUnit(
            channels_in=input_channels_mult
                        * channels_net
                        * channels_expansion**(i+1),
            channels_out=channels_net * channels_expansion**i,
            groups=groups,
            batch_norm=batch_norm,
            dropout=dropout,
            bias=bias)
                    for i in reversed(range(depth-1))]

        self.synthesis_track = nn.ModuleList(up_track)

        # Final class prediction
        self.predict = nn.Sequential(
            nn.Conv2d(input_channels_mult * channels_net, channels_net, 3, 1,
                      1,
                      1,
                      1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, channels_net, 3, 1, 1, 1,
                      1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, 1, 1, 1, 0, 1,
                      1,
                      bias=bias))

        self.apply(initialize_weights)

    def forward(self, x, x_brg):
        fx = self.embedding(x)

        for layer, bridge_layer, x_k in zip(self.synthesis_track
                                            + [self.predict],
                                            reversed(x_brg)):
            fx = torch.cat((fx, x_k), dim=1)
            fx = layer(fx)

        return fx
    
class ConvNet(nn.Module):
    """ConvNet model for end-to-end phenotype synthesis.
    """
    def __init__(self, channels_org=3, classes=1, channels_net=8,
                 depth=3,
                 channels_expansion=1,
                 groups=False,
                 batch_norm=False,
                 dropout=0.5,
                 bias=True,
                 use_bridge=True,
                 **kwargs):
        super(ConvNet, self).__init__()

        self.analysis = Analyzer(channels_org, channels_net, depth,
                                 channels_expansion,
                                 groups,
                                 batch_norm,
                                 dropout,
                                 bias)
        
        self.synthesis = Synthesizer(classes, channels_net,
                                     depth-1,
                                     channels_expansion,
                                     groups,
                                     batch_norm,
                                     dropout,
                                     use_bridge=use_bridge,
                                     
                                     bias=bias)
        self.use_bridge

    def forward(self, img, img_mask=None):
        fimg, fimg_brg = self.analysis(img)
        if self.use_bridge:
            fimg_brg=fimg_brg[:-1]
        else:
            fimg_brg=None
        fimg = self.synthesis(fimg, fimg_brg)
        if img_mask is None:
            fimg=torch.mean(fimg,(2,3))
        else:
            out=torch.zeros((fimg.shape[0],1))
            for ii in range(fimg.shape[0]):
                fimg_sub=fimg[ii,0,:,:]
                out[ii]=torch.mean(fimg_sub[img_mask[ii,:,:]])
            syth_pheno=out
        return syth_pheno

    def extract_features(self, x, fx_brg=None):
        fx, fx_brg = self.analysis(x)
        fx = self.bottleneck(fx)
        y, fx = self.synthesis.extract_features(fx, fx_brg[:-1])
        return y, fx


class BNNet(nn.Module):
    def __init__(self, depth, channels_bn, channels_net, channels_expansion=1, bias=True, groups=False, batch_norm=True, dropout=0.5):
        super(BNNet, self).__init__()
        self.embedding = nn.ConvTranspose2d(
            channels_bn,
            channels_net
            * channels_expansion**(depth-1),
            kernel_size=2,
            stride=2,
            padding=0,
            groups=channels_bn if groups else 1,
            bias=bias)
        up_track = [UpsamplingUnit(
            channels_in=channels_net
                        * channels_expansion**(i+1),
            channels_out=channels_net * channels_expansion**i,
            groups=groups,
            batch_norm=batch_norm,
            dropout=dropout,
            bias=bias)
                    for i in reversed(range(depth-1))]
        self.synthesis_track = nn.Sequential(*up_track)
        self.predict = nn.Sequential(
            nn.Conv2d(channels_net, channels_net, 3, 1, 1, 1, 1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, channels_net, 3, 1, 1, 1, 1,
                      bias=bias,
                      padding_mode='reflect'),
            nn.Conv2d(channels_net, 1, 1, 1, 0, 1, 1,
                      bias=bias))
        self.apply(initialize_weights)
        
    def forward(self,x,bn_mask=None):
        fx=self.embedding(x)
        fx=self.synthesis_track(fx)
        fx=self.predict(fx)
        if bn_mask is None:
            fx=torch.mean(fx,(2,3))
        else:
            out=torch.zeros((fx.shape[0],1))
            for ii in range(fx.shape[0]):
                fx_sub=fx[ii,0,:,:]
                out[ii]=torch.mean(fx_sub[bn_mask[ii,:,:]])
            fx=out
        return fx

class LinierKinship(nn.Module):
    def __init__(self,N):
        super(LinierKinship, self).__init__()
        self.linier=torch.nn.Linear(N, 1, bias=True)
    def forward(self,K):
        fk=self.linier(K)
        return fk
    
    
    
class BNLinDeepCCA(nn.Module):
    def __init__(self, depth, channels_bn, channels_net, N=None, channels_expansion=1, bias=True, groups=False, batch_norm=True, dropout=0.5,**kwargs):
        super(BNLinDeepCCA, self).__init__()
        if N is None:
            raise TypeError('ConvLinDeepCCA needs keyword-only argument N')
        self.bn_model=BNNet(depth, channels_bn, channels_net, channels_expansion, bias, groups, batch_norm, dropout)
        self.kinship_model=LinierKinship(N)
        self.loss=cca_loss().loss
        
        
    def forward(self,bn,K,bn_mask=None,**kwargs):
        o_bn=self.bn_model(bn,bn_mask)
        o_k=self.kinship_model(K)
        return o_bn, o_k
    
class BNBNDeepCCA(nn.Module):
    def __init__(self, depth, channels_bn, channels_net, channels_expansion=1, bias=True, groups=False, batch_norm=True, dropout=0.5,**kwargs):
        super(BNBNDeepCCA, self).__init__()
        self.bn_1_model=BNNet(depth, channels_bn, channels_net, channels_expansion, bias, groups, batch_norm, dropout)
        self.bn_2_model=BNNet(depth, channels_bn, channels_net, channels_expansion, bias, groups, batch_norm, dropout)
        self.loss=cca_loss().loss
        
        
    def forward(self,bn_1,bn_2, bn_1_mask=None, bn_2_mask=None,**kwargs):
        o_bn_1=self.bn_1_model(bn_1)
        o_bn_2=self.bn_2_model(bn_2)
        return o_bn_1, o_bn_2        
        
class ConvLinDeepCCA(nn.Module):
    def __init__(self, depth=3, channels_net=128, N=None, channels_expansion=2, bias=True, groups=False, batch_norm=True, dropout=0.5, use_bridges=True,**kwargs):
        super(BNLinDeepCCA, self).__init__()
        if N is None:
            raise TypeError('ConvLinDeepCCA needs keyword-only argument N')
        self.img_model=ConvNet(depth, channels_net, channels_expansion, bias, groups, batch_norm, dropout, use_bridges)
        self.kinship_model=LinierKinship(N)
        self.loss=cca_loss().loss
        
        
    def forward(self,img,K,img_mask=None,**kwargs):
        o_pheno=self.img_model(img,img_mask)
        o_geno=self.kinship_model(K)
        return o_pheno, o_geno
    
class ConvConvDeepCCA(nn.Module):
    def __init__(self, depth, channels_net, channels_expansion=1, bias=True, groups=False, batch_norm=True, dropout=0.5,use_bridges=True,**kwargs):
        super(ConvConvDeepCCA, self).__init__()
        self.img_1_model=BNNet(depth, channels_net, channels_expansion, bias, groups, batch_norm, dropout, use_bridges)
        self.img_2_model=BNNet(depth, channels_net, channels_expansion, bias, groups, batch_norm, dropout, use_bridges)
        self.loss=cca_loss().loss
        
        
    def forward(self,img_1,img_2, img_1_mask=None, img_2_mask=None,**kwargs):
        o_pheno_1=self.img_1_model(img_1)
        o_pheno_2=self.img_2_model(img_2)
        return o_pheno_1, o_pheno_2
    
        
        