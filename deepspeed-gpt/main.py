import os
import logging
import time
import torch
import numpy as np
from transformers import pipeline
import deepspeed
import graphsignal

logging.basicConfig(level=logging.DEBUG)

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(deployment='gpt-neo-prod', debug_mode=True)

local_rank = int(os.getenv('LOCAL_RANK', '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))

generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B', device=local_rank)

config = {
    'tensor_parallel': {
        'tp_size': 2
    },
    'dtype': 'fp16',
    'replace_method': "auto",
    'replace_with_kernel_inject': True
}

ds_engine = deepspeed.init_inference(generator.model, config=config)
generator.model = ds_engine.module

local_rank = deepspeed.comm.get_local_rank()

try:
    while True:
        input_text = 'DeepSpeed is'
        # Graphsignal: measure inference
        with graphsignal.start_trace(endpoint='predict') as trace:
            trace.set_data('input', input_text)
            output = generator(input_text, do_sample=False, min_length=50, max_length=50)
            trace.set_data('output', output)
        time.sleep(1)
except KeyboardInterrupt:
    print('exiting...')
