#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install langchain
pip install numexpr
pip install openai
pip install graphsignal

deactivate
