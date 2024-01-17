#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade langchain
pip install --upgrade numexpr
pip install --upgrade openai
pip install --upgrade graphsignal

deactivate
