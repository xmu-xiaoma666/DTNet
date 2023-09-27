import torch
from torch import nn
import numpy as np

class sMLPBlock(nn.Module):
    def __init__(self,h=7,w=7,c=512):
        super().__init__()
        self.proj_h=nn.Linear(h,h)
        self.proj_w=nn.Linear(w,w)
        self.fuse=nn.Linear(3*c,c)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,input):
        x=input
        bs,n,dim=x.shape
        h,w=int(np.sqrt(n)),int(np.sqrt(n))
        x=x.view(bs,h,w,dim)
        x_h=self.proj_h(x.permute(0,3,2,1)).permute(0,3,2,1)
        x_w=self.proj_w(x.permute(0,1,3,2)).permute(0,1,3,2)
        x_id=x
        x_fuse=torch.cat([x_h,x_w,x_id],dim=-1)
        att=self.sigmoid(self.fuse(x_fuse).view(bs,-1,dim))
        out=input*att
        return out