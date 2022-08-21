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
            res = requests.post('http://localhost:8001/predict', json={'text': 'This API is so good'})
            logger.debug('Prediction output: %s', res.json())
        except:
            logger.error('Prediction request failed', exc_info=True)
        
        time.sleep(random.randint(1, 5) * 0.1)

t = threading.Thread(target=send_requests)
t.start()