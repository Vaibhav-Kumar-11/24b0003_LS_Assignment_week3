# (i) Load the IMDb dataset using the datasets library
from datasets import load_dataset
dataset = load_dataset("imdb")





# (ii) Preprocess the dataset, including tokenization using the appropriate tokenizer for bert-base-uncased.
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True) # this implies tokenizer to entire dataset






# (iii) Fine-tune the BERT model for sentiment analysis (binary classification: positive or negative).
from transformers import BertForSequenceClassification, TrainingArguments, Trainer
import torch

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Now let's define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs"
)

# Formatting the dataset for PyTorch
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Split up the datasets
train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(5000))  # Subset for speed
eval_dataset = tokenized_datasets["test"].select(range(1000))

# Initializing the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

trainer.train() # Being obvious it trains the model







# (iv) Evaluate the model’s performance using accuracy and F1-score metrics.
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    labels = pred.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

trainer.compute_metrics = compute_metrics
metrics = trainer.evaluate()
print(metrics)

# (v) Save the fine-tuned model and demonstrate how to load it for inference on a sample text input.

# Saving the fine tuned-model
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

# Loading the model
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="./sentiment_model", tokenizer="./sentiment_model")

# Inference on a sample text input as asked in the question
print(sentiment_pipeline("This movie was absolutely wonderful!"))
