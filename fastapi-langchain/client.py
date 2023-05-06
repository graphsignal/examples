import logging
import time
import random
import threading
import requests

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def send_requests():
    while True:
        try:
            logger.debug('Sending request')

            idea = 'colorful socks'

            # simulate inference exception
            if random.random() < 0.001:
                idea = None

            # simulate missing value
            if random.random() < 0.001:
                idea = ''

            res = requests.post('http://localhost:8001/generate', json=dict(idea=idea))
            logger.debug('Generated output: %s', res.json())
        except:
            logger.error('Request failed', exc_info=True)

        time.sleep(random.randint(1, 5))

t = threading.Thread(target=send_requests)
t.start()