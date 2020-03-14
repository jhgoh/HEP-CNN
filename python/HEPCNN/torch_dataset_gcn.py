#!/usr/bin/env pythnon
import torch
#from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import Dataset as InMemoryDataset
from torch_geometric.data import Data as PyGData
from HEPCNN.torch_dataset import HEPCNNDataset
from HEPCNN.torch_dataset_splited import HEPCNNSplitDataset
import numpy as np
from math import pi, cos, sin, sinh

"""@torch.jit.script
def fillNodeInfo(idxs, image, poses0):
    np = len(idxs)
    poses = torch.zeros([np, 3])#, requires_grad=False)
    feats = torch.zeros([np, 3])#, requires_grad=False)

    for i, idx in enumerate(idxs):
        poses[i] = poses0[idx[0],idx[1]]
        feats[i] = image[:,idx[0],idx[1]]

    return poses, feats
"""

class HEPGCNDataset(InMemoryDataset):
#class HEPGCNDataset(PyGDataset, HEPCNNSplitDataset):
    def __init__(self, dirName, nEvent=-1, syslogger=None):
        super(HEPGCNDataset, self).__init__('/')
        if dirName.endswith('.h5'):
            self.dataset = HEPCNNDataset(dirName, nEvent, syslogger=syslogger)
        else:
            self.dataset = HEPCNNSplitDataset(dirName, nEvent, nWorkers=8, syslogger=syslogger)
        self.width = self.dataset.width
        self.height = self.dataset.height

        ## The image will be NCHW, [batch][detector][phi][eta]
        phis = np.arange(0, 2*pi, 2*pi/self.width)+(2*pi/self.width/2)
        etas = np.arange(-2.5, 2.5, (2.5+2.5)/self.height)+(2.5+2.5)/self.height/2
        self.poses0 = torch.zeros([self.width, self.height, 3], requires_grad=False)
        r0 = 1.317 ## scale radius to have 2*r0*sin(dphi/2) = deta. r0=1.317 for deta=2.5 and r0->1 for small deta
        for i, phi in enumerate(phis):
            for j, eta in enumerate(etas):
                x, y = r0*cos(phi), r0*sin(phi)
                z = eta
                self.poses0[i,j] = torch.tensor([x,y,z], requires_grad=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        pass

    def get(self, idx):
        data = []
        image, label, weight = self.dataset.__getitem__(idx)
        label = label.to(torch.long)

        ## Build nodes, remove empty points
        idxs = torch.unique(torch.nonzero(image)[:,1:], dim=0)
        np = len(idxs)
        poses = torch.zeros([np, 3], requires_grad=False)
        feats = torch.zeros([np, 3], requires_grad=False)
        for i, idx in enumerate(idxs):
            poses[i] = self.poses0[idx[0],idx[1],]
            feats[i] = image[:,idx[0],idx[1]]
        #poses, feats = fillNodeInfo(idxs, image, self.poses0)

        data = PyGData(x=feats, pos=poses, y=label.item())
        #data = PyGData(x=image.permute(1,2,0).view(-1,3), pos=self.poses, y=label.item())
        data.weight = weight.item()

        return data

    def __len__(self):
        return len(self.dataset)
