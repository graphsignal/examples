import logging
import graphsignal

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Graphsignal: import and configure
#   expects GRAPHSIGNAL_API_KEY environment variable
graphsignal.configure()
tracer = graphsignal.tracer(with_profiler='pytorch')

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

class MyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def prediction_step(self, *args, **kwargs):
        # Graphsignal: measure and profile inference
        with tracer.trace(endpoint='bert-imdb') as span:
            return super().prediction_step(*args, **kwargs)


from transformers import Trainer
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset)

trainer.train()

trainer.evaluate()
