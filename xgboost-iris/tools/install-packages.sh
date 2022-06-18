#!/bin/bash

set -e 

python3 -m venv venv
source venv/bin/activate

pip install sklearn
pip install xgboost
pip install graphsignal

deactivate
