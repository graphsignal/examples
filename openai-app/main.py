import os
import logging
import time
import logging
import random
import openai
import graphsignal

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

openai.api_key = os.getenv('OPENAI_API_KEY')

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(deployment='openai-app-example')


prompts = [
  'What is quantum computing and how does it differ from classical computing?',
  'What are qubits and how are they different from classical bits?',
  'How does quantum computing use quantum mechanics to perform computations?',
  'What is quantum entanglement and how is it used in quantum computing?',
  'What is quantum teleportation and how does it relate to quantum computing?',
  'How are quantum algorithms different from classical algorithms?',
  'What is quantum parallelism and how does it impact computational speed?',
  'What are some common quantum algorithms and what are they used for?',
  'What are the current limitations and challenges of quantum computing?',
  'What are some potential applications and implications of quantum computing in the future?'
]

while True:
    try:
        response = openai.Completion.create(
            model="text-davinci-003", 
            prompt=prompts,
            temperature=0.7,
            top_p=1,
            max_tokens=1024,
            frequency_penalty=0,
            presence_penalty=0)
        logger.debug("Response: %s", response)
    except openai.error.RateLimitError as rle:
        logger.debug("Rate limit error: %s", rle)

    time.sleep(5 * random.random())
