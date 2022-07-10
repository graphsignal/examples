import logging
# Graphsignal: import
import graphsignal
from graphsignal.profilers.pytorch import profile_inference

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: import and configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure(workload_name='Hugging Face BERT IMDB')


from datasets import load_dataset
raw_datasets = load_dataset("imdb")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"],
                        padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(100))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]

from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

from transformers import Trainer, TrainingArguments
training_args = TrainingArguments("test_trainer")

# Graphsignal: profiler prediction_step
class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def prediction_step(self, *args, **kwargs):
        with profile_inference(batch_size=training_args.eval_batch_size):
            return super().prediction_step(*args, **kwargs)


from transformers import Trainer
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset)

trainer.train()

trainer.evaluate()
