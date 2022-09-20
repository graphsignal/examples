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
graphsignal.configure()
tracer = graphsignal.tracer(with_profiler='pytorch')

app = FastAPI()

pipe = pipeline(task="text-classification", model="distilbert-base-uncased", device=0)

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()

    # Graphsignal: measure and profile inference
    print('request_id', body['request_id'])
    with tracer.span(endpoint='distilbert-prod', tags=dict(request_id=body['request_id'])) as span:
        span.set_data('input', body['text'])
        output = pipe(body['text'])
    return JSONResponse(content={"output": output})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")