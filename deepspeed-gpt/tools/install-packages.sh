#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install torch
pip install torchmetrics
pip install torchvision
pip install transformers
pip install datasets
pip install deepspeed
pip install graphsignal

deactivate
