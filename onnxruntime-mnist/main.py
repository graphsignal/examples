import logging
import os
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelPruning, QuantizationAwareTraining
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
import onnxruntime

# Graphsignal: import
import graphsignal
from graphsignal.profilers.onnxruntime import initialize_profiler, profile_inference

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: import and configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(workload_name='ONNX Runtime MNIST')

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

TEST_MODEL_PATH = 'temp/mnist.onnx'

if not os.path.exists(TEST_MODEL_PATH):
    os.mkdir('temp')
    
    class MNISTModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(28 * 28, 10)
            self.batch_size = BATCH_SIZE
            self.train_acc = Accuracy()

        def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
            optimizer.zero_grad(set_to_none=True)

        def forward(self, x):
            return torch.relu(self.l1(x.view(x.size(0), -1)))

        def training_step(self, batch, batch_nb):
            x, y = batch
            preds = self(x)
            loss = F.cross_entropy(preds, y)
            self.train_acc(preds, y)
            self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
            return loss

        def train_dataloader(self):
            train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
            train_loader = DataLoader(train_ds, batch_size=self.batch_size)
            return train_loader

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=0.02)

    mnist_model = MNISTModel()

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=10
    )
    trainer.fit(mnist_model)

    input_sample = torch.randn((mnist_model.batch_size, 28 * 28))
    mnist_model.to_onnx(
        TEST_MODEL_PATH, 
        input_sample, 
        export_params=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},
                        'output' : {0 : 'batch_size'}})

sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
# Graphsignal: initialize profiler for ONNX inference session.
initialize_profiler(sess_options)

session = onnxruntime.InferenceSession(TEST_MODEL_PATH, sess_options)

test_ds = MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

test_acc = Accuracy()
for x, y in test_loader:
    # Graphsignal: measure and profile inference.
    with profile_inference(session):
        preds = session.run(None, { 'input': x.detach().cpu().numpy().reshape((x.shape[0], 28 * 28)) })
        test_acc.update(torch.tensor(preds[0]), y)

# Graphsignal: log test accuracy.
graphsignal.log_metric('test_acc', test_acc.compute().item())
