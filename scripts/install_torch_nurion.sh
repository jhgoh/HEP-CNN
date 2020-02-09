#!/bin/bash

source /apps/compiler/intel/19.0.5/impi/2019.5.281/intel64/bin/mpivars.sh release_mt
source /apps/applications/miniconda3/etc/profile.d/conda.sh
module load git craype-mic-knl
module load gcc/8.3.0
export USE_CUDA=0
export USE_MKLDNN=1
export USE_OPENMP=1
export USE_TBB=0

## Just for once
rm -rf /scratch/$USER/conda/nurion_torch
conda create -p /scratch/$USER/conda/nurion_torch
##### Proceed ([y]/n)? y

## Every time
conda activate /scratch/$USER/conda/nurion_torch

## Just for once
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing
conda install -y h5py tqdm scikit-learn matplotlib pandas
##### Proceed ([y]/n)? y

cd /scratch/$USER
git clone --recursive https://github.com/pytorch/pytorch -b v1.4.0
##### Takes long time to download linked packages?

cd pytorch
git submodule sync
git submodule update --init --recursive
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
##### This takes even longer time to compile SW?
cd ..
