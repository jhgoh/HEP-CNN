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

class PointDynamicEdgeConv(PyG.EdgeConv):
    def __init__(self, nn, k, aggr='max', **kwargs):
        super(PointDynamicEdgeConv, self).__init__(nn=nn, aggr=aggr, **kwargs)
        self.k = k

    def forward(self, x, pos, batch=None):
        edge_index = PyG.knn_graph(pos, self.k, batch, loop=False, flow=self.flow)
        return super(PointDynamicEdgeConv, self).forward(x, edge_index)

    def __repr__(self):
        return '{}(nn={}, k={})'.format(self.__class__.__name__, self.nn,
                                        self.k)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.nChannel = 3

        self.conv1 = PointDynamicEdgeConv(MLP([self.nChannel+3, 64, 128]), 8, 'max')
        self.conv2 = PyG.DynamicEdgeConv(MLP([2 * 128, 256]), 4, 'max')
        self.conv3 = PyG.DynamicEdgeConv(MLP([2 * 256, 512]), 4, 'max')

        self.lin1 = MLP([512 + 256 + 128, 1024])

        self.fc = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear( 512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x, pos, batch = data.x, data.pos, data.batch
        x1 = self.conv1(x, pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = PyG.global_max_pool(out, batch)
        out = self.fc(out)
        return out
