import os
import logging
import time
import torch
import numpy as np
from transformers import pipeline
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import graphsignal

logging.basicConfig(level=logging.DEBUG)

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(deployment='app-prod')

app = FastAPI()

pipe = pipeline(task="text-classification", model="distilbert-base-uncased")

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()

    # Graphsignal: measure and profile inference
    print('request_id', body['request_id'])
    with graphsignal.start_trace(endpoint='distilbert-prod', tags=dict(request_id=body['request_id']), profiler='pytorch') as trace:
        trace.set_data('input', body['text'])
        output = pipe(body['text'])
    return JSONResponse(content={"output": output})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
