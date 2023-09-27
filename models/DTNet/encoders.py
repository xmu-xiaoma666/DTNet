from numpy.core.shape_base import stack
from torch.nn import functional as F
from torch.nn.modules.activation import GELU
from .utils import PositionWiseFeedForward
import torch
from torch import nn
from .attention import MultiHeadAttention
from ..relative_embedding import GridRelationalEmbedding
from .seblock import SEPerceptron
from .repblobk import LocalPerceptron
from .smlpblock import sMLPBlock
import numpy as np
import math

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Router(nn.Module):
    def __init__(self,n=49,dim=512,r1=7,r2=16,path=3):
        super().__init__()
        self.spatial_router=nn.Sequential(
            nn.Linear(n,n//r1),
            nn.ReLU(),
            nn.Linear(n//r1,path),
        )

        self.channel_router=nn.Sequential(
            nn.Linear(dim,dim//r2),
            nn.ReLU(),
            nn.Linear(dim//r2,path),
        )

        self.finnal_router=nn.Sequential(
            nn.Linear(path*2,path),
            nn.ReLU(),
            nn.Linear(path,path),
            nn.Softmax(dim=-1)
        )

    def forward(self,x):
        bs,n,dim=x.shape
        channel_info=x.mean(1) #bs,dim
        spatial_info=x.mean(2) #bs,n
        spatial_weight=self.spatial_router(spatial_info) #bs,path
        channel_weight=self.channel_router(channel_info) #bs,path
        weight=torch.cat([channel_weight,spatial_weight],dim=-1) #bs,path*2
        weight=self.finnal_router(weight).unsqueeze(-1).unsqueeze(-1) #bs,path,1,1
        return weight


class SpatialMultiBranch(nn.Module):
    def __init__(self,d_model=512,dropout=0.1):
        super().__init__()
        # self.identity=nn.Identity()

        self.lp=LocalPerceptron(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)


        self.smlp=sMLPBlock(h=7,w=7,c=512)
        self.dropout4 = nn.Dropout(dropout)
        self.lnorm4 = nn.LayerNorm(d_model)


    def forward(self,x,sa_att):
        bs,n,dim = x.shape
        # att1=self.identity(x)
        att2=self.lnorm2(x+self.dropout2(self.lp(x)))
        att4=self.lnorm4(x+self.dropout4(self.smlp(x)))
        att=torch.stack([att2,att4,sa_att],1) #bs,path,n,dim
        return att



class ChannelMultiBranch(nn.Module):
    def __init__(self,d_model=512,dropout=0.1):
        super().__init__()
        # self.identity=nn.Identity()

        self.se=SEPerceptron(channel=d_model,reduction=16)
        self.dropout3 = nn.Dropout(dropout)
        self.lnorm3 = nn.LayerNorm(d_model)

    def forward(self,x,ff_att):
        bs,n,dim = x.shape
        # att1=self.identity(x)
        att3=self.lnorm3(x+self.dropout3(self.se(x)))
        att=torch.stack([att3,ff_att],1) #bs,path,n,dim
        return att



class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

        self.spatialmultibranch=SpatialMultiBranch(d_model=d_model)
        self.channelmultibranch=ChannelMultiBranch(d_model=d_model)
        self.spatialrouter=Router(path=3)
        self.channelrouter=Router(path=2)
        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.lnorm3 = nn.LayerNorm(d_model)

    def forward(self, queries, keys, values, relative_pos=None, attention_mask=None, attention_weights=None, pos=None):
        #Self-Attention
        sa_att = self.mhatt(queries, queries, queries, relative_pos, attention_mask, attention_weights)
        sa_att = self.lnorm(queries + self.dropout(sa_att))

        #spatial dynamic net
        all_att=self.spatialmultibranch(queries,sa_att)
        all_weight=self.spatialrouter(queries)
        sp_att=torch.sum(all_att*all_weight,1)
        sp_att = self.lnorm2(queries + self.dropout2(sp_att))



        #FFN
        ff = self.pwff(sp_att)

        #channel dynamic net
        all_att=self.channelmultibranch(sp_att,ff)
        all_weight=self.channelrouter(sp_att)
        ch_att=torch.sum(all_att*all_weight,1)
        ch_att = self.lnorm3(sp_att + self.dropout3(ch_att))



        return ch_att


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx
        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, input, attention_weights=None, pos=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input == 0, -1) != 0).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        relative_geometry_embeddings = GridRelationalEmbedding(input.shape[0])
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [w(flatten_relative_geometry_embeddings).view(box_size_per_head) for w in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        grid2grid = F.relu(relative_geometry_weights)
        out = input #bs,n,dim
        for l in self.layers:
            out = l(out, out, out, grid2grid, attention_mask, attention_weights, pos=pos)

        return out, attention_mask


from numpy import math
import numpy as np
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        rowPE = torch.zeros(max_len,max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        rowPE[ :,:,0::2] = torch.sin(position * div_term)
        rowPE[ :,:, 1::2] = torch.cos(position * div_term)
        colPE=rowPE.transpose(1, 0)
        rowPE = rowPE.unsqueeze(0)
        colPE = colPE.unsqueeze(0)
        self.rowPE=rowPE.cuda()
        self.colPE=colPE.cuda()

    def forward(self, x):
        feat=x
        bs,gs,dim=feat.shape
        feat=feat.view(bs,int(np.sqrt(gs)),int(np.sqrt(gs)),dim)
        feat = feat + self.rowPE[:, :int(np.sqrt(gs)), :int(np.sqrt(gs)),  :dim ]+ self.colPE[:,  :int(np.sqrt(gs)),  :int(np.sqrt(gs)),  :dim ]
        feat=feat.view(bs,-1,dim)
        return self.dropout(feat)


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.pe=PositionalEncoding(d_model=d_in,dropout=0)

    def forward(self, input, attention_weights=None):
        feat=self.pe(input)
        mask = (torch.sum(feat, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(feat))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)


