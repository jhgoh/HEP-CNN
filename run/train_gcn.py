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
resourceByCPFile = os.path.join(args.outdir, 'resourceByCP_%d.csv' % hvd_rank)
resourceByTimeFile = os.path.join(args.outdir, 'resourceByTime_%d.csv' % hvd_rank)

proc = subprocess.Popen(['python', '../scripts/monitor_proc.py', '-t', '1',
                        '-o', resourceByTimeFile, '%d' % os.getpid()],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

import time
class TimeHistory():#tf.keras.callbacks.Callback):
    def on_train_begin(self):
        self.times = []
    def on_epoch_begin(self):
        self.epoch_time_start = time.time()
    def on_epoch_end(self):
        self.times.append(time.time() - self.epoch_time_start)

sys.path.append("../scripts")
from monitor_proc import SysStat
sysstat = SysStat(os.getpid(), fileName=resourceByCPFile)
sysstat.update(annotation="start_loggin")

sys.path.append("../python")
from HEPCNN.torch_dataset import HEPGCNDataset as MyDataset

sysstat.update(annotation="open_trn")
trnDataset = MyDataset(args.trndata, args.ntrain)
sysstat.update(annotation="read_trn")

sysstat.update(annotation="open_val")
valDataset = MyDataset(args.valdata, args.ntest)
sysstat.update(annotation="read_val")

kwargs = {'num_workers':min(4, nthreads)}
if torch.cuda.is_available() and hvd:
    kwargs['num_workers'] = 1
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

if hvd:
    compression = hvd.Compression.none
    #compression = hvd.Compression.fp16 #if args.fp16_allreduce else hvd.Compression.none
    optm = hvd.DistributedOptimizer(optm,
                                    named_parameters=model.named_parameters(),
                                    compression=compression, backward_passes_per_step=args.batch)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optm, root_rank=0)

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

sysstat.update(annotation="modelsetup_done")

from tqdm import tqdm
from sklearn.metrics import accuracy_score
bestModel, bestAcc = {}, -1
if not os.path.exists(weightFile):
    try:
        timeHistory = TimeHistory()
        timeHistory.on_train_begin()
        sysstat.update(annotation="train_start")
        history = {'time':[], 'loss':[], 'acc':[], 'val_loss':[], 'val_acc':[]}

        for epoch in range(args.epoch):
            timeHistory.on_epoch_begin()
            sysstat.update(annotation='epoch_begin')

            model.train()
            trn_loss, trn_acc = 0., 0.
            loss = None
            for i, (data, label, weight) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, args.epoch))):
                #data = data.to(device)
                #weight = weight.float()

                optm.zero_grad()
                pred = model(data).float()
                crit = torch.nn.BCELoss()#weight=weight)
                l = crit(pred.view(-1), label.float()).to('cpu')
                l.backward()
                if loss is None: loss = l
                else: loss += l
                if i % args.batchPerStep == 0 or i+1 == len(trnLoader):
                    #loss.backward()
                    optm.step()

                trn_loss += l.item()
                trn_acc += accuracy_score(label, np.where(pred > 0.5, 1, 0))

                sysstat.update()
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

            timeHistory.on_epoch_end()
            sysstat.update(annotation='epoch_end')
            history['loss'].append(trn_loss)
            history['acc'].append(trn_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

        sysstat.update(annotation="train_end")

        history['time'] = timeHistory.times[:]
        with open(trainingFile, 'w') as f:
            writer = csv.writer(f)
            keys = history.keys()
            writer.writerow(keys)
            for row in zip(*[history[key] for key in keys]):
                writer.writerow(row)
        sysstat.update(annotation="wrote_logs")

    except KeyboardInterrupt:
        print("Training finished early")

if hvd_rank == 0:
    torch.save(bestModel, weightFile)

    model.load_state_dict(torch.load(weightFile))
    model.eval()
    #pred = model(valDataset.images.to(device))

    #np.save(predFile, pred.to('cpu').detach().numpy())
    sysstat.update(annotation="saved_model")
