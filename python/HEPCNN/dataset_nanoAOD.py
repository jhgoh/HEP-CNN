#!/usr/bin/env pythnon
import numpy as np
#import h5py
import uproot
import awkward
import torch
#from torch.utils.data import Dataset
from bisect import bisect_right
from os import listdir, environ
import concurrent.futures as futures

from torch_geometric.data import Dataset as InMemoryDataset
from torch_geometric.data import Data as PyGData
from math import pi, cos, sin, sinh

#@torch.jit.script
#def etaphi2xyz(eta, phi):
#    r0 = 1.0
#    x, y = r0*cos(phi), r0*sin(phi)
#    z = r0*sinh(eta)
#    return x, y, z

class NanoAODDataset(InMemoryDataset):
    def __init__(self, dirName, nEvent=-1, **kwargs):
        super(NanoAODDataset, self).__init__('/')
        syslogger = kwargs['syslogger'] if 'syslogger' in kwargs else None
        nWorkers = kwargs['nWorkers'] if 'nWorkers' in kwargs else 8

        if syslogger: syslogger.update(annotation='open file '+ dirName)
        self.dirName = dirName
        self.maxEventsList = [0,]
        self.files = []
        self.trees = []
        self.fileIdx = -1

        if syslogger: syslogger.update(annotation='read files')

        nEventsTotal = 0
        for fileName in sorted(listdir(self.dirName)):
            if not fileName.endswith('.root'): continue
            f = uproot.open(self.dirName+'/'+fileName)
            tree = f['Events']
            #print([x for x in tree.keys() if 'Weight' in str(x)])
            #print('\n'.join([str(x) for x in tree.keys() if 'HLT' not in str(x)]))
            self.files.append(f)
            self.trees.append(tree)

            nEventsInFile = len(tree)
            nEventsTotal += nEventsInFile
            self.maxEventsList.append(nEventsTotal)

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
        fileIdx = bisect_right(self.maxEventsList, idx)-1

        if self.fileIdx != fileIdx:
            self.fileIdx = fileIdx
            tree = self.trees[fileIdx]
            weights = tree.arrays(["genWeight", "LHEWeight_originalXWGTUP"], outputtype=tuple)
            self.weights = weights[0]/weights[1]
            jetDauEtas, jetDauPhis = tree.arrays(["jetDau_eta", "jetDau_phi"], outputtype=tuple)
            #jetDauVars = tree.arrays(["jetDau_pt", "jetDau_E", "jetDau_charge", "jetDau_pdgId"], outputtype=tuple)
            jetDauVars = tree.arrays(["jetDau_pt", "jetDau_E", "jetDau_charge"], outputtype=tuple)

            xs, ys = np.cos(jetDauPhis), np.sin(jetDauPhis)
            zs = np.sinh(jetDauEtas)
            self.poses = awkward.JaggedArray.zip(xs, ys, zs)
            self.feats = awkward.JaggedArray.zip(*jetDauVars)

        offset = self.maxEventsList[fileIdx]
        idx = idx - offset

        feat = torch.tensor(self.feats[idx].tolist())
        pos  = torch.tensor(self.poses[idx].tolist())
        data = PyGData(x=feat, pos=pos, y=self.label)
        data.weight = float(self.weights[idx])

        return data

    def __len__(self):
        return self.maxEventsList[-1]


