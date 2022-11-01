# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 09:22:13 2022

@author: mawes
"""

import torch

class cca_loss():
    def loss(self,o_x1,o_x2):
        x1_std,x1_mean=torch.std_mean(o_x1,unbiased=False)
        x2_std,x2_mean=torch.std_mean(o_x2,unbiased=False)
        corr=torch.matmul(((o_x1-x1_mean)/x1_std).T,(o_x2-x2_mean)/x2_std)
        corr=corr/o_x1.size(0)
        corr=-corr**2
        return corr