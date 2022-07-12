#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install wheel
pip install torch
#pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchmetrics
pip install torchvision
pip install graphsignal

deactivate
