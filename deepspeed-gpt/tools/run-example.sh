#!/bin/bash

set -e 

source venv/bin/activate
deepspeed --num_gpus 2 main.py
deactivate