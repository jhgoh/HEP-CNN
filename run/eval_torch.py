#!/usr/bin/env python
import numpy as np
import argparse
import sys, os
import subprocess
import csv
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

nthreads = int(os.popen('nproc').read()) ## nproc takes allowed # of processes. Returns OMP_NUM_THREADS if set
torch.set_num_threads(nthreads)

parser = argparse.ArgumentParser()
parser.add_argument('--batch', action='store', type=int, default=256, help='Batch size')
parser.add_argument('-t', '--test', action='store', type=str, required=True, help='Test dataset')
parser.add_argument('-d', '--input', action='store', type=str, required=True, help='directory with pretrained model parameters')

args = parser.parse_args()

predFile = args.input+'/prediction.csv'
import pandas as pd

if not os.path.exists(predFile):
    sys.path.append("../python")

    #from HEPCNN.torch_dataset import HEPCNNSplitedDataset as MyDataset
    from HEPCNN.torch_dataset import HEPCNNDataset as MyDataset

    kwargs = {'num_workers':min(4, nthreads), 'pin_memory':True}

    testDataset = MyDataset(args.test)
    testLoader = DataLoader(testDataset, batch_size=args.batch, shuffle=False, **kwargs)

    from HEPCNN.torch_model_default import MyModel
    model = MyModel(testDataset.width, testDataset.height)

    device = 'cpu'
    if torch.cuda.is_available():
        model = model.cuda()
        device = 'cuda'

    from tqdm import tqdm
    model.load_state_dict(torch.load(args.input+'/weight_0.pkl'))
    model.eval()

    labels, preds = [], []
    for i, (data, label, weight) in enumerate(tqdm(testLoader)):
        data = data.float().to(device)
        weight = weight.float()
        pred = model(data).detach().to('cpu').float()

        labels.extend([x.item() for x in label])
        preds.extend([x.item() for x in pred.view(-1)])
    df = pd.DataFrame({'label':labels, 'prediction':preds})
    df.to_csv(predFile, index=False)

from sklearn.metrics import roc_curve, roc_auc_score
df = pd.read_csv(predFile)
tpr, fpr, thr = roc_curve(df['label'], df['prediction'], pos_label=0)
auc = roc_auc_score(df['label'], df['prediction'])

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='%s %.3f' % (args.input, auc))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim(0, 0.001)
plt.ylim(0, 1.000)
plt.legend()
plt.show()
