#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install wheel
pip install openai
pip install tiktoken
pip install langchain
pip install chromadb
pip install graphsignal

deactivate
