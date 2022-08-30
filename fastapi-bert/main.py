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
from graphsignal.tracers.pytorch import inference_span

logging.basicConfig(level=logging.DEBUG)

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure()

app = FastAPI()

pipe = pipeline(task="text-classification", model="distilbert-base-uncased", device=0)

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()

    # Graphsignal: measure and profile inference
    print('request_id', body['request_id'])
    s2 = time.perf_counter()
    with inference_span(model_name='distilbert-prod', tags=dict(request_id=body['request_id'])):
        s1 = time.perf_counter()
        output = pipe(body['text'])
        print('S1', time.perf_counter() - s1)
    print('S2', time.perf_counter() - s2)
    return JSONResponse(content={"output": output})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")