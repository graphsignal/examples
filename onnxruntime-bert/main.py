import uuid
from pathlib import Path
from transformers import AutoTokenizer, pipeline, PretrainedConfig
from optimum.onnxruntime import ORTModelForSequenceClassification

import argparse
parser = argparse.ArgumentParser(description='Benchmark args')
parser.add_argument('--onnx', action='store_true')
parser.add_argument('--gpu', action='store_true')
parser.add_argument('--quantize', dest='quantize', action='store_true')
args = parser.parse_args()


model_id="optimum/distilbert-base-uncased-finetuned-banking77"
dataset_id="banking77"
onnx_path = Path("/tmp/onnx")
task = "text-classification"

payload = "What can I do if my card still hasn't arrived after 2 weeks?"

def compute_accuracy(pipe):
    print('Evaluating...')

    from evaluate import evaluator
    from datasets import load_dataset 

    eval = evaluator("text-classification")
    eval_dataset = load_dataset("banking77", split="test")

    results = eval.compute(
        model_or_pipeline=pipe,
        data=eval_dataset,
        metric="accuracy",
        input_column="text",
        label_column="label",
        label_mapping=pipe.model.config.label2id,
        strategy="simple",
    )
    return results['accuracy']


# Graphsignal: configure, expects GRAPHSIGNAL_API_KEY environment variable
import graphsignal
graphsignal.configure(deployment='distilbert')


if not args.onnx:
    vanilla_clx = pipeline(task, model=model_id, device=0 if args.gpu else -1)
    accuracy = compute_accuracy(vanilla_clx)

    for _ in range(100):
        # Graphsignal: measure and profile inference
        with graphsignal.start_trace(endpoint='predict-vanilla'):
            _ = vanilla_clx(payload)

if args.onnx:
    if args.quantize:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        quantizer = ORTQuantizer.from_pretrained(model_id, feature=model.pipeline_task)
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)

        quantizer.export(
            onnx_model_path=onnx_path / "model.onnx",
            onnx_quantized_model_output_path=onnx_path / "model.onnx",
            quantization_config=qconfig,
        )
    else:
        model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
        model.save_pretrained(onnx_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(onnx_path)

    import onnxruntime

    from graphsignal.profilers.onnxruntime_profiler import ONNXRuntimeProfiler()
    ort_profiler = ONNXRuntimeProfiler()

    sess_options = onnxruntime.SessionOptions()

    # Graphsignal: initialize profiler for ONNX Runtime session
    ort_profiler.initialize_options(sess_options)

    session = onnxruntime.InferenceSession(
        str(onnx_path / 'model.onnx'),
        sess_options,
        providers=[
            'CUDAExecutionProvider' if args.gpu else 'CPUExecutionProvider'
        ])
    ort_profiler.set_onnx_session(session)

    model_from_session = ORTModelForSequenceClassification(
        model=session, 
        config=PretrainedConfig.from_json_file(onnx_path / 'config.json'))

    optimum_clx = pipeline(task, model=model_from_session, tokenizer=tokenizer, device=0 if args.gpu else -1)
    accuracy = compute_accuracy(optimum_clx)

    for _ in range(100):
        # Graphsignal: measure and profile inference
        with graphsignal.start_trace(
                endpoint='predict-onnx', 
                tags=dict(run_name='run1'),
                profiler=ort_profiler):
            _ = optimum_clx(payload)