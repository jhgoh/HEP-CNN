#!/usr/bin/env pythnon
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

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

