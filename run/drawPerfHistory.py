#!/usr/bin/env python
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import argparse
from collections import OrderedDict

metrics_all = ['CPU', 'RSS', 'VMSize', 'Read', 'Write']
metrics_opts = {'CPU':('%', 1), 'RSS':('GB',1e9), 'VMSize':('GB',1e9), 'Read':('MB',1e6), 'Write':('MB',1e6)}

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--metrics', type=str, action='store', default='CPU',
                    help="List of metrics (CPU,RSS,VMSize,Read,Write,all)")
parser.add_argument('dirs', type=str, nargs='+', action='store',
                    help="Directories with log messages")
parser.add_argument('-a', '--annotation', action='store_true', default=False,
                    help="Add annotations")
args = parser.parse_args()

metrics = []
for metric in args.metrics.split(','):
    if metric == 'all':
        metrics = metrics_all
        break
    if metric not in metrics_all: continue
    metrics.append(metric)

dirs = []
for d in args.dirs:
    if not os.path.exists(d+'/resourceByCP_0.csv') and \
       not os.path.exists(d+'/resourceByCP_0.csv.gz'): continue
    dirs.append(d)

cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

#plt.rcParams['lines.linewidth'] = 1
#plt.rcParams['lines.markersize'] = 5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams['lines.markersize'] = 2

plt.rcParams['figure.figsize'] = (7, len(metrics)*2)
maxTime = 0

data = OrderedDict()
for d in dirs:
    print(d)
    for metric in metrics:
        if metric not in data: data[metric] = []

        data[metric].append([])

        for ii in range(64):
            suffix = ".csv"
            if not os.path.exists('%s/resourceByCP_%d.csv' % (d, ii)):
                if os.path.exists('%s/resourceByCP_%d.csv.gz' % (d, ii)): suffix = ".csv.gz"
                else: continue
            usage1 = pd.read_csv('%s/resourceByCP_%d%s' % (d, ii, suffix))
            usage2 = pd.read_csv('%s/resourceByTime_%d%s' % (d, ii, suffix))
            usage = usage1.append(usage2, ignore_index=True)

            usage['Datetime'] = pd.to_datetime(usage['Datetime'], format='%Y-%m-%d %H-%M-%S')
            usage = usage.sort_values(['Datetime'])
            beginTime = min(usage['Datetime'])

            usage['time'] = (usage['Datetime']-beginTime).dt.total_seconds()
            maxTime = max(max(usage['time']), maxTime)
            scale = metrics_opts[metric][1]
            usage[metric] /= scale

            data[metric][-1].append(usage)

if len(dirs) > 1 or len(metrics) == 1:
    plt.rcParams['figure.figsize'] = (7, len(dirs)*2)
    for metric in metrics:
        for i, d in enumerate(dirs):
            hostInfo, pars = d.split('/',1)
            hostAlias, hostSpec = hostInfo.replace('perf_', '').split('_',1)

            ax = plt.subplot(len(dirs), 1, i+1)
            if maxTime > 0: ax.set_xlim([0, maxTime*1.1])
            plt.title(pars.replace('__', ' '))

            for usage in data[metric][i]:
                plt.plot(usage['time'], usage[metric], '.-', alpha=0.5)

                if args.annotation:
                    for t, m, s in zip(usage['time'], usage[metric], usage['Annotation']):
                        if type(s) != str: continue
                        s = ' '.join(s.split()[:-1]) if len(s.split()) > 2 else s
                        plt.text(t, m, s)

            plt.grid(linestyle=':')
            if i == len(dirs)-1: plt.xlabel('time')
            unit = metrics_opts[metric][0]
            plt.ylabel('%s(%s)' % (metric, unit))

            #plt.legend()

        plt.tight_layout()
        plt.savefig('%s.png' % (metric))
        plt.show()
        #    ax.set_ylim([0, 6000])

if len(metrics) > 1:
    plt.rcParams['figure.figsize'] = (7, len(metrics)*2)
    for j, d in enumerate(dirs):
        for i, metric in enumerate(metrics):
            hostInfo, pars = d.split('/',1)
            hostAlias, hostSpec = hostInfo.replace('perf_', '').split('_',1)

            ax = plt.subplot(len(metrics), 1, i+1)
            #if maxTime > 0: ax.set_xlim([0, maxTime*1.1])
            #ax.set_xlim([0, 3000])

            for usage in data[metric][j]:
                plt.plot(usage['time'], usage[metric], '.-', alpha=0.5)#, label=('rank%d'%ii))#, c=cols[i], label=(pars.replace('__', ' ')))

                if args.annotation:
                    for t, m, s in zip(usage['time'], usage[metric], usage['Annotation']):
                        if type(s) != str: continue
                        s = ' '.join(s.split()[:-1]) if len(s.split()) > 2 else s
                        plt.text(t, m, s)

            plt.grid(linestyle=':')
            if i == len(metrics)-1: plt.xlabel('time')
            unit = metrics_opts[metric][0]
            plt.ylabel('%s(%s)' % (metric, unit))

            #plt.yscale('log')
            #if metric == 'CPU':
            #    ax.set_ylim([0, 6000])
            #plt.legend()

        plt.tight_layout()
        plt.savefig('%s/%s.png' % (d, metric))
        plt.show()

