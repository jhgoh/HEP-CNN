#!/usr/bin/env python
import torch
import torch.nn as nn

import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch.conv import GraphConv

class MyModel(nn.Module):
    def __init__(self, width, height, nchannel=3):
        super(MyModel, self).__init__()

        self.gc1 = GraphConv(3, 64, activation=F.relu)
        self.gc2 = GraphConv(64, 256, activation=F.relu)
        self.gc3 = GraphConv(256, 256, activation=F.relu)

        self.fc = nn.Sequential(
            nn.Linear(64*width*height,1),
            #nn.Linear(width*height,1),
            nn.Sigmoid(),
        )

    def forward(self, grps):
        x = []
        for g in dgl.unbatch(grps):
            hcal, ecal, trck = g.ndata['hcal'], g.ndata['ecal'], g.ndata['trck']
            xx = torch.cat([hcal, ecal, trck]).view(3,-1).permute(1,0)
            xx = self.gc1(g, xx)
            x.append(xx)
        x = torch.cat(x).view(grps.batch_size,-1)

        #batch_size = grps.batch_size
        #hcals, ecals, trcks = grps.ndata['hcal'], grps.ndata['ecal'], grps.ndata['trck']
        #l = hcals.shape[-1]//batch_size
        #x = torch.cat([hcals, ecals, trcks]) ## this makes CN(HW)
        #x = x.view(3,-1,l).permute(1,0,2) ## convert back to NC(HW)
        #x = x.reshape(-1,3*l) ## convert back to NC(HW)
        #print(x.view(-1).shape)

        #x = self.gc1(grps, hcals.view(-1))
        #print(x.shape)
        x = self.fc(x)

        return x
