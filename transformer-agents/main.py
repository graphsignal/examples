import logging
import time
import random
from transformers.tools import OpenAiAgent

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# import and initialize graphsignal
# add GRAPSIGNAL_API_KEY to your environment variables
import graphsignal
graphsignal.configure(deployment='transofmer-agents-example')


agent = OpenAiAgent(model="text-davinci-003")


def translate(text):
    agent.run(f'Translate the following text to French: {text}', remote=True)


# simulate some requests
texts=[
    'Software installation complete',
    'System update available',
    'Network error',
    'Access denied',
    'Backup in progress'
]

while True:
    num = random.randint(0, len(texts) - 1)
    
    try:
        translation = translate(texts[num])
        logger.debug(f'Translation: {translation}')
    except:
        logger.error("Error while translating text", exc_info=True)

    time.sleep(5 * random.random())
