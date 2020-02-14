#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as PyG

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PyG.PointConv(nn)

    def forward(self, x, pos, batch):
        idx = PyG.fps(pos, batch, ratio=self.ratio)
        row, col = PyG.radius(pos, pos[idx], self.r, batch, batch[idx],
                              max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = PyG.global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

def MLP(channels, batch_norm=True):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU(), nn.BatchNorm1d(channels[i]))
        for i in range(1, len(channels))
    ])

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.sa1 = SAModule(0.5, 0.2, MLP([3+2, 64, 64, 128]))
        self.sa2 = SAModule(0.25, 0.4, MLP([128 + 2, 128, 128, 256]))
        self.sa3 = GlobalSAModule(MLP([256 + 2, 256, 512, 1024]))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        sa1_out = self.sa1(data.x, data.pos, data.batch)
        sa2_out = self.sa2(*sa1_out)
        sa3_out = self.sa3(*sa2_out)
        x, pos, batch = sa3_out

        x = self.fc(x)

        return x
