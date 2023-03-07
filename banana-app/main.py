import os
import logging
import time
import logging
import random
import banana_dev as banana
import graphsignal

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(deployment='banana-example-client')


while True:
    model_inputs = {
        "prompt": "Hello World! I am a [MASK] machine learning model."
    }
    out = banana.run(os.environ['BANANA_API_KEY'], 'b69a2292-1197-4ef3-b076-5b361d0efa74', model_inputs)        
    logger.debug("Response: %s", out)

    time.sleep(15 * random.random())
