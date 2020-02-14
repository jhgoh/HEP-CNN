#!/usr/bin/env pythnon
import torch
#from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import Dataset as InMemoryDataset
from torch_geometric.data import Data as PyGData
from HEPCNN.torch_dataset_splited import HEPCNNSplitDataset

class HEPGCNDataset(InMemoryDataset):
#class HEPGCNDataset(PyGDataset, HEPCNNSplitDataset):
    def __init__(self, dirName, nEvent=-1, syslogger=None):
        super(HEPGCNDataset, self).__init__('/')
        self.dataset = HEPCNNSplitDataset(dirName, nEvent, nWorkers=8, syslogger=syslogger)
        self.width = self.dataset.width
        self.height = self.dataset.height

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
        nx, ny = image.shape[-1], image.shape[-2] ## NCHW
        poses = torch.zeros([nx*ny, 2])
        feats = torch.zeros([nx*ny, 3]) ## 3 channel
        k = 0
        for iy in range(ny):
            for ix in range(nx):
                poses[k][0], poses[k][1] = ix, iy
                for ii, feat in enumerate(image[:,iy,ix]):
                    feats[k][ii] = feat
                k += 1
        data = PyGData(x=feats, pos=poses, y=label.item())
        data.weight = weight.item()
        return data

    def __len__(self):
        return len(self.dataset)
