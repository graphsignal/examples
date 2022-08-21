#!/bin/bash

set -e 

source venv/bin/activate

python -m transformers.onnx --model=distilbert-base-uncased --feature=sequence-classification temp/

deactivate