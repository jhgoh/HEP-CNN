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

try:
    import horovod.torch as hvd
except:
    hvd = None

nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
print("NTHREADS=", nthreads, "CPU_COUNT=", os.cpu_count())
torch.set_num_threads(nthreads)

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
parser.add_argument('--shuffle', action='store', type=bool, default=True, help='Shuffle batches for each epochs')
parser.add_argument('--optimizer', action='store', choices=('sgd', 'adam', 'radam', 'ranger'), default='adam', help='optimizer to run')
parser.add_argument('--model', action='store', choices=('default', 'log3ch', 'log5ch', 'original', 'circpad', 'circpadlog3ch', 'circpadlog5ch'), 
                               default='default', help='choice of model')

args = parser.parse_args()

hvd_rank, hvd_size = 0, 1
if hvd:
    hvd.init()
    hvd_rank = hvd.rank()
    hvd_size = hvd.size()
    print("Horovod is available. (rank=%d size=%d)" % (hvd_rank, hvd_size))
    #torch.manual_seed(args.seed)
    #torch.cuda.set_device(hvd.local_rank())

if not os.path.exists(args.outdir): os.makedirs(args.outdir)
modelFile = os.path.join(args.outdir, 'model.pkl')
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
from HEPCNN.torch_dataset_splited import HEPCNNSplitDataset as MySplitDataset
from HEPCNN.torch_dataset import HEPCNNDataset as MyDataset

sysstat.update(annotation="open_trn")
if os.path.isdir(args.trndata):
    trnDataset = MySplitDataset(args.trndata, args.ntrain, nWorkers=nthreads//2, syslogger=sysstat)
else:
    trnDataset = MyDataset(args.trndata, args.ntrain, syslogger=sysstat)
sysstat.update(annotation="read_trn")

sysstat.update(annotation="open_val")
if os.path.isdir(args.valdata):
    valDataset = MySplitDataset(args.valdata, args.ntest, nWorkers=nthreads//2, syslogger=sysstat)
else:
    valDataset = MyDataset(args.valdata, args.ntest, syslogger=sysstat)
sysstat.update(annotation="read_val")

kwargs = {'num_workers':min(4, nthreads)}
#if torch.cuda.is_available() and hvd:
#    kwargs['num_workers'] = 1
kwargs['pin_memory'] = True

if hvd:
    trnSampler = torch.utils.data.distributed.DistributedSampler(trnDataset, num_replicas=hvd_size, rank=hvd_rank)
    valSampler = torch.utils.data.distributed.DistributedSampler(valDataset, num_replicas=hvd_size, rank=hvd_rank)
    trnLoader = DataLoader(trnDataset, batch_size=args.batch, sampler=trnSampler, **kwargs)
    valLoader = DataLoader(valDataset, batch_size=args.batch, sampler=valSampler, **kwargs)
else:
    trnLoader = DataLoader(trnDataset, batch_size=args.batch, shuffle=args.shuffle, **kwargs)
    #valLoader = DataLoader(valDataset, batch_size=args.batch, shuffle=args.shuffle, **kwargs)
    valLoader = DataLoader(valDataset, batch_size=512, shuffle=False, **kwargs)

## Build model
if args.model == 'original':
    from HEPCNN.torch_model_original import MyModel
elif 'circpad' in args.model:
    from HEPCNN.torch_model_circpad import MyModel
else:
    from HEPCNN.torch_model_default import MyModel
model = MyModel(trnDataset.width, trnDataset.height, model=args.model)
if hvd_rank == 0: torch.save(model, modelFile)
device = 'cpu'
if torch.cuda.is_available():
    model = model.cuda()
    device = 'cuda'

if args.optimizer == 'radam':
    from optimizers.RAdam import RAdam
    optm = RAdam(model.parameters(), lr=args.lr)
elif args.optimizer == 'ranger':
    from optimizers.RAdam import RAdam
    from optimizers.Lookahead import Lookahead
    optm_base = RAdam(model.parameters(), lr=args.lr)
    optm = Lookahead(optm_base)
elif args.optimizer == 'adam':
    optm = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'sgd':
    optm = optim.SGD(model.parameters(), lr=args.lr)
else:
    print("Cannot find optimizer in the list")
    exit()

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

with open(args.outdir+'/summary.txt', 'w') as fout:
    fout.write(str(args))
    fout.write('\n\n')
    fout.write(str(model))
    fout.close()

from tqdm import tqdm
from sklearn.metrics import accuracy_score
bestModel, bestAcc = {}, -1
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
        optm.zero_grad()
        for i, (data, label, weight) in enumerate(tqdm(trnLoader, desc='epoch %d/%d' % (epoch+1, args.epoch))):
            data = data.float().to(device)
            label = label.float().to(device)
            weight = weight.float().to(device)

            pred = model(data)
            crit = torch.nn.BCELoss(weight=weight)
            if device == 'cuda': crit = crit.cuda()
            l = crit(pred.view(-1), label)
            l.backward()
            if i % args.batchPerStep == 0 or i+1 == len(trnLoader):
                optm.step()
                optm.zero_grad()

            trn_loss += l.item()
            trn_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0))

            sysstat.update()
        trn_loss /= len(trnLoader)
        trn_acc  /= len(trnLoader)

        model.eval()
        val_loss, val_acc = 0., 0.
        for i, (data, label, weight) in enumerate(tqdm(valLoader)):
            data = data.float().to(device)
            label = label.float().to(device)
            weight = weight.float().to(device)

            pred = model(data)
            crit = torch.nn.BCELoss(weight=weight)
            loss = crit(pred.view(-1), label)

            val_loss += loss.item()
            val_acc += accuracy_score(label.to('cpu'), np.where(pred.to('cpu') > 0.5, 1, 0))
        val_loss /= len(valLoader)
        val_acc  /= len(valLoader)

        if hvd: val_acc = metric_average(val_acc, 'avg_accuracy')
        if bestAcc < val_acc:
            bestModel = model.state_dict()
            bestAcc = val_acc
            if hvd_rank == 0:
                torch.save(bestModel, weightFile)
                sysstat.update(annotation="saved_model")

        timeHistory.on_epoch_end()
        sysstat.update(annotation='epoch_end')
        history['loss'].append(trn_loss)
        history['acc'].append(trn_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        history['time'].append(timeHistory.times[-1])
        if hvd_rank == 0:
            with open(trainingFile, 'w') as f:
                writer = csv.writer(f)
                keys = history.keys()
                writer.writerow(keys)
                for row in zip(*[history[key] for key in keys]):
                    writer.writerow(row)
            sysstat.update(annotation="wrote_logs")

    sysstat.update(annotation="train_end")

except KeyboardInterrupt:
    print("Training finished early")

