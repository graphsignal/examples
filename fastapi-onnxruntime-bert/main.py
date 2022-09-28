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

logging.basicConfig(level=logging.DEBUG)

# Graphsignal: configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure()

from graphsignal.profilers.onnxruntime_profiler import ONNXRuntimeProfiler()
ort_profiler = ONNXRuntimeProfiler()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", cache_dir='temp/cache')

sess_options = onnxruntime.SessionOptions()
ort_profiler.initialize_options(sess_options)

session = onnxruntime.InferenceSession("temp/model.onnx", sess_options, providers=['CUDAExecutionProvider'])
ort_profiler.set_onnx_session(session)

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    inputs = tokenizer(body['text'], return_tensors="np")
    # Graphsignal: measure and profile inference
    with graphsignal.start_trace(endpoint='DistilBERT-prod-gpu', profiler=ort_profiler):
        outputs = session.run(output_names=["logits"], input_feed=dict(inputs))
    return JSONResponse(content={"outputs": outputs[0].tolist()})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")