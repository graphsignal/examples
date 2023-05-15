#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install openai
pip install transformers[agents]
pip install graphsignal

deactivate
