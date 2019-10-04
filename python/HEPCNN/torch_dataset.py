#!/usr/bin/env pythnon
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import dgl
from dgl import DGLGraph
from copy import deepcopy

class HEPCNNDataset(Dataset):
    def __init__(self, fileName, nEvent=-1, syslogger=None):
        super(HEPCNNDataset, self).__init__()
        if syslogger: syslogger.update(annotation='open file '+ fileName)
        self.fileName = fileName
        if fileName.endswith('h5'):
            data = h5py.File(fileName, 'r')
        elif fileName.endswith('npz'):
            data = {'all_events':np.load(fileName)}
        suffix = "_val" if 'images_val' in data['all_events'] else ""

        if syslogger: syslogger.update(annotation='read file')
        self.images = data['all_events']['images'+suffix]
        self.labels = data['all_events']['labels'+suffix]
        self.weights = data['all_events']['weights'+suffix]

        if nEvent > 0:
            self.images = self.images[:nEvent]
            self.labels = self.labels[:nEvent]
            self.weights = self.weights[:nEvent]
        else:
            self.images = self.images[()]
            self.labels = self.labels[()]
            self.weights = self.weights[()]
        if syslogger: syslogger.update(annotation='select events')

        self.images = torch.Tensor(self.images)
        self.labels = torch.Tensor(self.labels)
        self.weights = torch.Tensor(self.weights)
        if syslogger: syslogger.update(annotation="Convert data to Tensors")

        self.shape = self.images.shape
        if self.shape[-1] <= 5:
            ## actual format was NHWC. convert to pytorch native format, NCHW
            self.images = self.images.permute(0,3,1,2)
            self.shape = self.images.shape
            if syslogger: syslogger.update(annotation="Convert image format")
        self.channel, self.height, self.width = self.shape[1:]

    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx], self.weights[idx])

    def __len__(self):
        return self.shape[0]

class HEPGCNDataset(HEPCNNDataset):
    def __init__(self, fileName, nEvent=-1, syslogger=None):
        super(HEPGCNDataset, self).__init__(fileName, nEvent, syslogger)
        width, height = self.images.shape[2:] ## Assume NCHW

        self.grptpl = dgl.DGLGraph()
        self.grptpl.add_nodes(width*height)
        for iy in range(height):
            for ix in range(width):
                i0 = iy*width + ix
                iwest = iy*height + ((ix-1+width) % width)
                ieast = iy*height + ((ix+1) % width)
                inorth = (iy-1) + ix ## inorth < 0 case will be skipped later
                isouth = (iy+1) + ix ## isouth > height case will be skipped later

                ## Self connection FIXME: do we need this?
                self.grptpl.add_edge(i0, i0)

                ## Connect east and west / left-right
                self.grptpl.add_edge(i0, iwest)
                self.grptpl.add_edge(i0, ieast)

                ## Connect north and south / up-down
                ## Skip the 0th and -1th to be a cylinder. otherwise, it will become a torus
                ## Note: for the zero-padding: should we add a 'bias' pixel with its content 0?
                if iy > 0 and iy < height-1:
                    self.grptpl.add_edge(i0, inorth)
                    self.grptpl.add_edge(i0, isouth)

    def collate(self, samples):
        grp, labels, weights = map(list, zip(*samples))
        batched_graph = dgl.batch(grp)
        labels = torch.Tensor(labels)
        weights = torch.Tensor(weights)
        
        return (batched_graph, labels, weights)
  
    def __getitem__(self, idx):
        grp = deepcopy(self.grptpl)
        image = self.images[idx]
        grp.ndata['hcal'] = image[0].view(-1)
        grp.ndata['ecal'] = image[1].view(-1)
        grp.ndata['trck'] = image[2].view(-1)

        return (grp, self.labels[idx], self.weights[idx])

