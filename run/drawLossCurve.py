#!/usr/bin/env python
import matplotlib.pyplot as plt
import pandas as pd
import sys, os

plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 5
plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["legend.frameon"] = False
plt.rcParams["legend.loc"] = 'upper left'

plt.rcParams['figure.figsize'] = (4*2, 3.5*3)
ax1 = plt.subplot(3, 2, 1, yscale='log', ylabel='Loss(train)', xlabel='epoch')
ax2 = plt.subplot(3, 2, 2, yscale='log', ylabel='Loss(val)', xlabel='epoch')
ax3 = plt.subplot(3, 2, 3, ylabel='Accuracy(train)', xlabel='epoch')
ax4 = plt.subplot(3, 2, 4, ylabel='Accuracy(val)', xlabel='epoch')
ax1.set_ylim([3e-2,1])
ax2.set_ylim([3e-2,1])
ax3.set_ylim([0.85,1])
ax4.set_ylim([0.85,1])
for ax in (ax1, ax2, ax3, ax4):
    ax.grid(which='major', axis='both', linestyle='-.')
    ax.grid(which='minor', linestyle=':')
    ax.set_xlim([0,50])
lines, labels = [], []
for d in sys.argv[1:]:
    df = pd.read_csv(d)

    label = d.split('/')[-2].replace('__', ' ').replace('_', '=')

    l = ax1.plot(df['loss'], '.-', label=label)
    ax2.plot(df['val_loss'], '.-', label=label)

    ax3.plot(df['acc'], '.-', label=label)
    ax4.plot(df['val_acc'], '.-', label=label)

    lines.append(l[0])
    labels.append(label)

ax5 = plt.subplot(3,1,3)
ax5.legend(lines, labels)
ax5.axis('off')

plt.tight_layout()
plt.show()
