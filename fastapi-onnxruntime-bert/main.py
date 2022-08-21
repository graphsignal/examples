import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer
import onnxruntime
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import graphsignal
from graphsignal.tracers.onnxruntime import initialize_profiler, inference_span

logging.basicConfig(level=logging.DEBUG)

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(workload_name='serving-prod-gpu')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir='temp/cache')

sess_options = onnxruntime.SessionOptions()

# Graphsignal: initialize session
initialize_profiler(sess_options)

session = onnxruntime.InferenceSession("temp/model.onnx", sess_options, providers=['CUDAExecutionProvider'])

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    inputs = tokenizer(body['text'], return_tensors="np")
    # Graphsignal: measure and profile inference
    with inference_span(model_name='DistilBERT', onnx_session=session):
        outputs = session.run(output_names=["logits"], input_feed=dict(inputs))
    return JSONResponse(content={"outputs": outputs[0].tolist()})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")