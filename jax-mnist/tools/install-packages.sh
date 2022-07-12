#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install wheel
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorflow
pip install tensorflow-datasets
pip install graphsignal

deactivate
