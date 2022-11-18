import logging
import time
import random
import threading
import requests

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def send_requests():
    request_id = 0
    while True:
        request_id += 1
        try:
            logger.debug('Sending request')

            text = 'This API is so good'

            # simulate inference exception
            if random.random() < 0.001:
                text = None

            # simulate missing value
            if random.random() < 0.001:
                text = ''

            res = requests.post('http://localhost:8001/predict', json=dict(request_id=request_id, text=text))
            logger.debug('Prediction output: %s', res.json())
        except:
            logger.error('Prediction request failed', exc_info=True)
        
        time.sleep(random.randint(1, 5) * 0.1)

t = threading.Thread(target=send_requests)
t.start()