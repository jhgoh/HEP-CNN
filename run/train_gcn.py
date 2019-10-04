#!/usr/bin/env python
import h5py
import numpy as np
import argparse
import sys, os
import subprocess
import csv
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', action='store', type=int, default=50, help='Number of epochs')
parser.add_argument('--ntrain', action='store', type=int, default=-1, help='Number of events for training')
parser.add_argument('--ntest', action='store', type=int, default=-1, help='Number of events for test/validation')
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('-t', '--trndata', action='store', type=str, required=True, help='input file for training')
parser.add_argument('-v', '--valdata', action='store', type=str, required=True, help='input file for validation')
parser.add_argument('-o', '--outdir', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--lr', action='store', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--noEarlyStopping', action='store_true', help='do not apply Early Stopping')
parser.add_argument('--batchPerStep', action='store', type=int, default=1, help='Number of batches per step (to emulate all-reduce)')

nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
print("NTHREADS=", nthreads, "CPU_COUNT=", os.cpu_count())
torch.set_num_threads(nthreads)

args = parser.parse_args()

hvd = None
hvd_rank = 0

if not os.path.exists(args.outdir): os.makedirs(args.outdir)
weightFile = os.path.join(args.outdir, 'weight_%d.pkl' % hvd_rank)
predFile = os.path.join(args.outdir, 'predict_%d.npy' % hvd_rank)
trainingFile = os.path.join(args.outdir, 'history_%d.csv' % hvd_rank)

sys.path.append("../python")
from HEPCNN.torch_dataset import HEPGCNDataset as MyDataset

trnDataset = MyDataset(args.trndata, args.ntrain)
valDataset = MyDataset(args.valdata, args.ntest)

kwargs = {'num_workers':min(4, nthreads)}
#if torch.cuda.is_available() and hvd:
#    kwargs['num_workers'] = 1
kwargs['pin_memory'] = True

trnLoader = DataLoader(trnDataset, batch_size=args.batch, shuffle=False, collate_fn=trnDataset.collate, **kwargs)
#valLoader = DataLoader(valDataset, batch_size=args.batch, shuffle=False, **kwargs)
valLoader = DataLoader(valDataset, batch_size=512, shuffle=False, collate_fn=valDataset.collate, **kwargs)

## Build model
from HEPCNN.torch_model_gcn import MyModel
model = MyModel(trnDataset.width, trnDataset.height)
#optm = optim.Adam(model.parameters(), lr=args.lr*hvd_size)
optm = optim.Adam(model.parameters(), lr=args.lr)

device = 'cpu'
#if torch.cuda.is_available():
#    model = model.cuda()
#    device = 'cuda'

from tqdm import tqdm
from sklearn.metrics import accuracy_score
bestModel, bestAcc = {}, -1
try:
    history = {'time':[], 'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}
    for epoch in range(args.epoch):
        model.train()
        trn_loss, trn_acc = 0., 0.
        loss = None
        for i, (data, label, weight) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, args.epoch))):
            #data = data.to(device)
            #weight = weight.float()

            optm.zero_grad()
            pred = model(data).float()
            crit = torch.nn.BCELoss()#weight=weight)
            l = crit(pred.view(-1), label)
            l.backward()
            if loss is None: loss = l
            else: loss += l
            if i % args.batchPerStep == 0 or i+1 == len(trnLoader):
                #loss.backward()
                optm.step()

            trn_loss += l.item()
            trn_acc += accuracy_score(label, np.where(pred > 0.5, 1, 0))

        trn_loss /= len(trnSampler) if hvd else (i+1)
        trn_acc  /= len(trnSampler) if hvd else (i+1)

        model.eval()
        val_loss, val_acc = 0., 0.
        for i, (data, label, weight) in enumerate(tqdm(valLoader)):
            data = data
            #weight = weight.float()

            pred = model(data)
            crit = torch.nn.BCELoss()#weight=weight)
            loss = crit(pred.view(-1), label.float())

            val_loss += loss.item()
            val_acc += accuracy_score(label, np.where(pred > 0.5, 1, 0))
        val_loss /= len(valSampler) if hvd else (i+1)
        val_acc  /= len(valSampler) if hvd else (i+1)

        if hvd: val_acc = metric_average(val_acc, 'avg_accuracy')
        if bestAcc < val_acc:
            bestModel = model.state_dict()
            bestAcc = val_acc

        history['loss'].append(trn_loss)
        history['acc'].append(trn_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    with open(trainingFile, 'w') as f:
        writer = csv.writer(f)
        keys = history.keys()
        writer.writerow(keys)
        for row in zip(*[history[key] for key in keys]):
            writer.writerow(row)
except KeyboardInterrupt:
    print("Training finished early")

