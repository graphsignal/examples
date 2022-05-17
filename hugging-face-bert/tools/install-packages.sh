#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

#pip install torch
pip install torch==1.11.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers
pip install datasets
pip install accelerate
pip install graphsignal

deactivate
