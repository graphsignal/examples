#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

if [ `uname -m` = "aarch64" ]; then
    export PIP_EXTRA_INDEX_URL=https://snapshots.linaro.org/ldcg/python-cache/ 
    pip install tensorflow-aarch64
else
    pip install tensorflow
fi
pip install tensorflow-datasets
pip install graphsignal

deactivate
