#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install wheel
if ! [ -x "$(command -v nvidia-smi)" ]; then
    pip install torch
else
    pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html
fi
pip install transformers
pip install datasets
if ! [ -x "$(command -v nvidia-smi)" ]; then
    pip install optimum[onnxruntime]
else
    pip install optimum[onnxruntime-gpu]
fi
pip install evaluate[evaluator]
pip install sklearn
pip install graphsignal

deactivate
