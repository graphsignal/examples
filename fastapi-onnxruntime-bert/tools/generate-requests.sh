#!/bin/bash

set -e 

source venv/bin/activate
python client.py
deactivate