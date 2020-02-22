#!/usr/bin/env python
import h5py
import numpy as np
import argparse
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', action='store', type=str, help='input file name', required=True)
parser.add_argument('-d', '--outdir', action='store', type=str, help='output file name', required=True)
parser.add_argument('-n', '--nevent', action='store', type=int, default=1024, help='number of events per file')
parser.add_argument('-c', '--chunk', action='store', type=int, default=1024, help='chunk size')
parser.add_argument('--nocompress', dest='nocompress', action='store_true', default=False, help='disable gzip compression')
args = parser.parse_args()

inFile = h5py.File(args.input, 'r')
suffix = '' if 'all_events/images' in inFile else '_val'
images  = inFile['all_events/images'+suffix]
labels  = inFile['all_events/labels'+suffix]
weights = inFile['all_events/weights'+suffix]
nEventsTotal = len(images)
chunkSize = min(args.chunk, args.nevent)

if not os.path.exists(args.outdir): os.makedirs(args.outdir)

for i, begin in enumerate(range(0, nEventsTotal, args.nevent)):
    end = min(begin+args.nevent, nEventsTotal)
    outFileName = '%s/%s_%d.h5' % (args.outdir, args.input.rsplit('/', 1)[1].rsplit('.', 1)[0], i)
    with h5py.File(outFileName, 'w', libver='latest') as outFile:
        print("Writing output file %s... [%d/%d]" % (outFileName, i+1, nEventsTotal//args.nevent))
        g = outFile.create_group('all_events')
        kwargs = {} if args.nocompress else {'compression':'gzip', 'compression_opts':9}
        image = images[begin:end]
        chunkSize = min(chunkSize, image.shape[0])
        g.create_dataset('images'+suffix, data=image, chunks=((chunkSize,)+image.shape[1:]), **kwargs)
        g.create_dataset('labels'+suffix, data=labels[begin:end], chunks=(chunkSize,))
        g.create_dataset('weights'+suffix, data=weights[begin:end], chunks=(chunkSize,))
        outFile.swmr_mode = True

