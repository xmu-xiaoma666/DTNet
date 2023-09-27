import torch
from torch import mean, nn
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from numpy import random
import os
# os.environ['CUDA_VISIBLE_DEVICES']='2'


class SEAttention(nn.Module):
    
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEPerceptron(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.se=SEAttention(channel=channel,reduction=reduction)
    
    def forward(self,x):
        bs,n,dim=x.shape
        h,w=int(np.sqrt(n)),int(np.sqrt(n))
        input=x.view(bs,h,w,dim).permute(0,3,1,2) #bs,dim,h,w
        out=self.se(input)
        out=out.reshape(bs,dim,-1).permute(0,2,1) #bs,n,dim
        return out
        