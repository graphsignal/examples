#!/bin/bash

set -e 

source venv/bin/activate
python train.py
deactivate