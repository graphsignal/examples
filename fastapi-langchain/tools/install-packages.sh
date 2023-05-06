#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install --upgrade setuptools
pip install wheel
pip install fastapi
pip install openai
pip install langchain
pip install uvicorn
pip install graphsignal

deactivate
