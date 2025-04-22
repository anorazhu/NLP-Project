import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from evaluate import load
from data_cleaning import ResumeDataProcessor

import os
os.environ["WANDB_DISABLED"] = "true"

# Load and preprocess data
processor = ResumeDataProcessor("data.csv")
_, _, _, _, _, _, df = processor.process()


# Combine all text features into one "text" column
feature_columns = df.columns[df.columns != 'job_category']
df['text'] = df[feature_columns].apply(
    lambda row: ' '.join(
        str(val).replace('\n', ', ').strip() 
        for val in row 
        if pd.notna(val) and str(val).strip().lower() != 'nan'
    ),
    axis=1
)

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['job_category'])

# Sampling with 100 
df = df.sample(n=100, random_state=42).reset_index(drop=True)

# Convert to Hugging Face dataset format
resume_dataset = Dataset.from_pandas(df[['text', 'label']])
resume_dataset = resume_dataset.train_test_split(test_size=0.2, seed=42)

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_resume = resume_dataset.map(preprocess_function)

# Padding and collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
num_labels = len(label_encoder.classes_)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Metrics
def compute_metrics(eval_pred):
   load_accuracy = load("accuracy")
   load_f1 = load("f1")

   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels, average="macro")["f1"]
   return {"accuracy": accuracy, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./bert_resume_results",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_resume['train'],
    eval_dataset=tokenized_resume['test'],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
trainer.evaluate()
print(trainer.evaluate())