#!/usr/bin/env python
from glob import glob
import matplotlib.pyplot as plt
import pandas as pd
import sys, os
import argparse
from collections import OrderedDict

labels = []
dtss = []
maxY = 0

for d in sys.argv[1:]:
    usage = pd.read_csv(d+'/resourceByCP_0.csv')
    usage['Datetime'] = pd.to_datetime(usage['Datetime'], format='%Y-%m-%d %H-%M-%S')
    usage = usage.sort_values(['Datetime'])
    dt1 = usage.query('Annotation == "epoch_begin"')['Datetime']
    dt2 = usage.query('Annotation == "epoch_end"')['Datetime']

    dts = [(t2-t1).seconds for (t1, t2) in zip(dt1, dt2)]
    dtss.append(dts)
    maxY = max(maxY, max(dts))

    labels.append(os.path.basename(d))
meandts = [sum(dts)/len(dts) for dts in dtss]

labels, meandts = zip(*list(sorted(zip(labels, meandts), key=lambda x: x[1])))

plt.rcParams['figure.figsize'] = (7, len(dtss)*0.5)
fig, ax = plt.subplots()
bar = ax.barh(range(len(labels)), meandts)
ax.set_yticks(range(len(labels)))
ax.set_xlabel('Second per epoch')
ax.set_yticklabels('')
for rect, label in zip(bar, labels):
    ax.text(rect.get_x()+1, rect.get_y()+rect.get_height()/2, label, ha='left', va='center')

plt.show()

