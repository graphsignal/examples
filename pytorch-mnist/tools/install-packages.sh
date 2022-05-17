#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install torch
#pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchmetrics
pip install torchvision
pip install graphsignal

deactivate
