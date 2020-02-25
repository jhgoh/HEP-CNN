#!/usr/bin/env pythnon
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from bisect import bisect_right
from os import listdir, environ
import concurrent.futures as futures

class HEPCNNSplitDataset(Dataset):
    def __init__(self, dirName, nEvent=-1, **kwargs):
        super(HEPCNNSplitDataset, self).__init__()
        syslogger = kwargs['syslogger'] if 'syslogger' in kwargs else None
        nWorkers = kwargs['nWorkers'] if 'nWorkers' in kwargs else 8

        if syslogger: syslogger.update(annotation='open file '+ dirName)
        self.dirName = dirName
        self.maxEventsList = [0,]
        self.imagesList = []
        self.labelsList = []
        self.weightsList = []
        self.fileIdx = -1

        if syslogger: syslogger.update(annotation='read files')

        nEventsTotal = 0
        for fileName in sorted(listdir(self.dirName)):
            if not fileName.endswith('h5'): continue
            data = h5py.File(self.dirName+'/'+fileName, 'r')
            suffix = "_val" if 'images_val' in data['all_events'] else ""

            images  = (fileName, 'all_events/images'+suffix) ## Keep the filename and image path only, and load them later with multiproc.
            #images = data['all_events/images'+suffix]
            labels  = data['all_events/labels'+suffix]
            weights = data['all_events/weights'+suffix]

            if nEvent > 0:
                #images  = images[:nEvent-nEventsTotal] ## We'll do this step after (re)loading the images
                labels  = labels[:nEvent-nEventsTotal]
                weights = weights[:nEvent-nEventsTotal]

            nEventsInFile = len(weights)
            nEventsTotal += nEventsInFile
            self.maxEventsList.append(nEventsTotal)

            labels  = torch.Tensor(labels[()])
            weights = torch.Tensor(weights[()])
            ## We will do this step for images later

            self.imagesList.append(images)
            self.labelsList.append(labels)
            self.weightsList.append(weights)

            if nEvent > 0 and nEventsTotal >= nEvent: break

        if syslogger: syslogger.update(annotation='Convert images to Tensor')

        env_kmp = environ['KMP_AFFINITY'] if 'KMP_AFFINITY' in environ else None
        environ['KMP_AFFINITY'] = 'none'
        jobs = []
        with futures.ProcessPoolExecutor(max_workers=nWorkers) as pool:
            for fileIdx in range(len(self.maxEventsList)-1):
                job = pool.submit(self.imageToTensor, fileIdx)
                jobs.append(job)

            for job in futures.as_completed(jobs):
                fileIdx, images = job.result()
                self.imagesList[fileIdx] = images
        if env_kmp != None: environ['KMP_AFFINITY'] = env_kmp

        for fileIdx in range(len(self.maxEventsList)-1):
            #images  = torch.Tensor(self.imagesList[fileIdx][()])
            images = self.imagesList[fileIdx]
            self.shape = images.shape

            if self.shape[-1] <= 5:
                ## actual format was NHWC. convert to pytorch native format, NCHW
                images = images.permute(0,3,1,2)
                self.shape = images.shape
                if syslogger: syslogger.update(annotation="Convert image format")

            self.imagesList[fileIdx] = images
            self.channel, self.height, self.width = self.shape[1:]

        if nEvent > 0:
            images  = images[:nEvent-nEventsTotal]

    def imageToTensor(self, fileIdx):
        fileName, imagesName = self.imagesList[fileIdx]
        data = h5py.File(self.dirName+'/'+fileName, 'r')
        images = data[imagesName]
        return fileIdx, torch.Tensor(images[()])

    def __getitem__(self, idx):
        fileIdx = bisect_right(self.maxEventsList, idx)-1
        offset = self.maxEventsList[fileIdx]

        if self.fileIdx != fileIdx:
            self.fileIdx = fileIdx

            self.images  = self.imagesList[fileIdx]
            self.labels  = self.labelsList[fileIdx]
            self.weights = self.weightsList[fileIdx]

        idx = idx - offset
        return (self.images[idx], self.labels[idx], self.weights[idx])

    def __len__(self):
        return self.maxEventsList[-1]

