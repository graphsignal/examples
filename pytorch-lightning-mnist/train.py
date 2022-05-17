import logging
import os
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
# Graphsignal: import
import graphsignal
from graphsignal.profilers.pytorch_lightning import GraphsignalCallback

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: import and configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(workload_name='PyTorch Lightning MNIST')

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64


class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.batch_size = BATCH_SIZE

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def train_dataloader(self):
        train_ds = MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
        train_loader = DataLoader(train_ds, batch_size=self.batch_size)
        return train_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

mnist_model = MNISTModel()


# Graphsignal: add profiler callback
trainer = Trainer(
    #auto_scale_batch_size=True,
    #accumulate_grad_batches=10,
    #auto_select_gpus=True,
    #accelerator='gpu',
    #devices=AVAIL_GPUS,
    #strategy="dp",
    gpus=2,
    accelerator='ddp',
    #precision=16,
    max_epochs=3,
    callbacks=[GraphsignalCallback(batch_size=mnist_model.batch_size)]
)

trainer.tune(mnist_model)

trainer.fit(mnist_model)