#!/bin/bash
module load gcc/7.2.0 openmpi

source /apps/applications/miniconda3/etc/profile.d/conda.sh
conda create --prefix /scratch/$USER/conda/pytorch
conda activate /scratch/$USER/conda/pytorch

conda install -y python=3.7
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing requests

conda install -y h5py scikit-learn pandas
conda install -y tqdm matplotlib
conda install -y -c pytorch pytorch-nightly-cpu
conda install -y -c intel openmp

HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod

