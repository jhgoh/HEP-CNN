#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as PyG

def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class PointConvNet(nn.Module):
    def __init__(self, net, r, **kwargs):
        super(PointConvNet, self).__init__()
        self.r = r
        self.conv = PyG.PointConv(net)

    def forward(self, x, pos, batch=None):
        #edge_index = PyG.knn_graph(pos, self.k, batch, loop=False, flow='source_to_target')
        edge_index = PyG.radius_graph(pos, self.r, batch, loop=False)
        x = self.conv(x, pos, edge_index)
        return x, pos, batch

class PoolingNet(nn.Module):
    def __init__(self, net):
        super(PoolingNet, self).__init__()
        self.net = net

    def forward(self, x, pos, batch):
        x = self.net(torch.cat([x, pos], dim=1))
        x = PyG.global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.nChannel = 3

        self.conv1 = PointConvNet(MLP([self.nChannel+3, 64, 64, 128]), 0.2)
        self.conv2 = PointConvNet(MLP([3+128, 128, 128, 256]), 0.2)
        self.pool = PoolingNet(MLP([256 + 3, 256, 512, 1024]))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear( 512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear( 256,   1),
            #nn.Sigmoid(),
        )

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x, pos, batch = self.conv1(x, pos, batch)
        x, pos, batch = self.conv2(x, pos, batch)
        x, pos, batch = self.pool(x, pos, batch)
        out = self.fc(x)
        return out
