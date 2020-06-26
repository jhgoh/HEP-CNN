#!/usr/bin/env python3
import h5py
import numpy as np
import argparse
import sys
from math import ceil

if sys.version_info[0] < 3: sys.exit()

parser = argparse.ArgumentParser()
parser.add_argument('input', nargs='+', action='store', type=str, help='input file names')
parser.add_argument('-o', '--output', action='store', type=str, help='output file name', required=True)
parser.add_argument('--nevent', action='store', type=int, default=-1, help='number of events to preprocess')
parser.add_argument('--nfiles', action='store', type=int, default=0, help='number of output files')
parser.add_argument('--format', action='store', choices=('NHWC', 'NCHW'), default='NHWC', help='image format for output (NHWC for TF default, NCHW for pytorch default)')
parser.add_argument('-c', '--chunk', action='store', type=int, default=1024, help='chunk size')
parser.add_argument('--nocompress', dest='nocompress', action='store_true', default=False, help='disable gzip compression')
parser.add_argument('-s', '--split', action='store_true', default=False, help='split output file')
parser.add_argument('-d', '--debug', action='store_true', default=False, help='debugging')
parser.add_argument('--precision', action='store', choices=(8,16,32,64), default=32, help='Precision')
args = parser.parse_args()

srcFileNames = [x for x in args.input if x.endswith('.h5')]
if not args.output.endswith('.h5'): outPrefix, outSuffix = args.output+'/data', '.h5'
else: outPrefix, outSuffix = args.output.rsplit('.', 1)
args.nevent = max(args.nevent, -1) ## nevent should be -1 to process everything or give specific value
precision = 'f%d' % (args.precision//8)
dtypes = {64:np.float64, 32:np.float32, 16:np.float16, 8:np.float8}
img_dtype = dtypes[args.precision]

## Logic for the arguments regarding on splitting
##   split off: we will simply ignore nfiles parameter => reset nfiles=1
##     nevent == -1: process all events store in one file
##     nevent != -1: process portion of events, store in one file
##   split on:
##     nevent == -1, nfiles == 1: same as the one without splitting
##     nevent != -1, nfiles == 1: same as the one without splitting
##     nevent == -1, nfiles != 1: process all events, split into nfiles
##     nevent != -1, nfiles != 1: split files, limit total number of events to be nevent
##     nevent != -1, nfiles == 0: split files by nevents for each files
if not args.split or args.nfiles == 1:
    ## Just confirm the options for no-splitting case
    args.split = False
    args.nfiles = 1
elif args.split and args.nevent > 0:
    args.nfiles = 0

## First scan files to get total number of events
print("@@@ Checking input files... (total %d files)" % (len(args.input)))
nEventTotal = 0
nEvent0s = []
for srcFileName in srcFileNames:
    data = h5py.File(srcFileName, 'r')['all_events']
    nEvent0 = data['hist'].shape[0]
    nEvent0s.append(nEvent0)
    nEventTotal += nEvent0
if args.nfiles > 0:
    nEventOutFile = int(ceil(nEventTotal/args.nfiles))
else:
    args.nfiles = int(ceil(nEventTotal/args.nevent))
    nEventOutFile = min(nEventTotal, args.nevent)
print("@@@ Total %d events to process, store into %d files (%d events per file)" % (nEventTotal, args.nfiles, nEventOutFile))

print("@@@ Start processing...")
outFileNames = []
nEventProcessed = 0
nEventToGo = nEventOutFile
out_labels, out_weights, out_image = None, None, None
for iSrcFile, (nEvent0, srcFileName) in enumerate(zip(nEvent0s, srcFileNames)):
    if args.debug: print("Open file", srcFileName)
    ## Open data file
    data = h5py.File(srcFileName, 'r')['all_events']

    weights = data['weight']
    labels = data['y'] if 'y' in data else np.ones(weights.shape[0])

    image_h = data['hist'][()].astype(img_dtype)
    image_e = data['histEM'][()].astype(img_dtype)
    image_t = data['histtrack'][()].astype(img_dtype)

    ## Preprocess image
    #image_e /= np.max(image_e)
    #image_t /= np.max(image_t)

    if args.debug and iSrcFile == 0:
        print("Build multi-channels image...")
        print("  Input image shape from the 1st file =", image_h.shape, image_e.shape, image_t.shape)

    if args.format == 'NHWC':
        image_h = np.expand_dims(image_h, -1)
        image_e = np.expand_dims(image_e, -1)
        image_t = np.expand_dims(image_t, -1)
        image = np.concatenate([image_h, image_e, image_t], axis=-1)
    else: ## for the NCHW
        image_h = np.expand_dims(image_h, 1)
        image_e = np.expand_dims(image_e, 1)
        image_t = np.expand_dims(image_t, 1)
        image = np.concatenate([image_h, image_e, image_t], axis=1)

    if args.debug and iSrcFile == 0:
        print("  Output image format=", args.format)
        print("  Output image shape from the 1st file =", image.shape)

    ## Put into the output file
    begin, end = 0, min(nEventToGo, nEvent0)
    while begin < nEvent0:
        ### First check to prepare output array
        if nEventToGo == nEventOutFile: ## Initializing output file
            ## Build placeholder for the output
            out_labels = np.ones(0)
            out_weights = np.ones(0)
            out_image = np.ones([0,*image.shape[1:]], dtype=img_dtype)
        ####

        ## Do the processing
        nEventToGo -= (end-begin)
        nEventProcessed += (end-begin)

        out_labels = np.concatenate([out_labels, labels[begin:end]])
        out_weights = np.concatenate([out_weights, weights[begin:end]])
        out_image = np.concatenate([out_image, image[begin:end,:,:,:]])

        begin, end = end, min(nEventToGo, nEvent0)

        if nEventToGo == 0 or nEventProcessed == nEventTotal: ## Flush output and continue
            nEventToGo = nEventOutFile
            end = min(begin+nEventToGo, nEvent0)

            iOutFile = len(outFileNames)+1
            outFileName = outPrefix + (("_%d" % iOutFile) if args.split else "") + ".h5"
            outFileNames.append(outFileName)
            if args.debug: print("Writing output file %s..." % outFileName, end='')

            chunkSize = min(args.chunk, out_weights.shape[0])
            with h5py.File(outFileName, 'w', libver='latest', swmr=True) as outFile:
                g = outFile.create_group('all_events')
                kwargs = {} if args.nocompress else {'compression':'gzip', 'compression_opts':9}
                kwargs['dtype'] = precision
                g.create_dataset('images', data=out_image, chunks=((chunkSize,)+out_image.shape[1:]), **kwargs)
                g.create_dataset('labels', data=out_labels, chunks=(chunkSize,), dtype='f4')
                g.create_dataset('weights', data=out_weights, chunks=(chunkSize,), dtype='f4')
                if args.debug: print("  done")

            with h5py.File(outFileName, 'r', libver='latest', swmr=True) as outFile:
                print(("  created %s (%d/%d)" % (outFileName, iOutFile, args.nfiles)), end='')
                print("  keys=", list(outFile.keys()), end='')
                print("  shape=", outFile['all_events']['images'].shape)

            continue

if args.debug:
    for outFileName in outFileNames:
        f = h5py.File(outFileName, 'r', libver='latest', swmr=True)
        print(outFileName, f['all_events/images'].shape)
