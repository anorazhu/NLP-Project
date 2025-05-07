import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
from evaluate import load
from data_cleaning import ResumeDataProcessor
from sklearn.model_selection import train_test_split
import os

os.environ["WANDB_DISABLED"] = "true"

# Load and preprocess data
processor = ResumeDataProcessor("data.csv")
X, y, X_train, X_test, y_train, y_test, df = processor.process()

# STEP 1: Combine text features into a single "text" column
if 'text' in df.columns:
    df = df.drop(columns=['text'])
    
X_df = df.drop(columns=["job_position_name", "job_category", "label"], errors='ignore').copy()
df['text'] = X_df.apply(
    lambda row: ' '.join(
        str(val).replace('\n', ', ').strip()
        for val in row
        if pd.notna(val) and str(val).strip().lower() != 'nan'
    ),
    axis=1
)
df['text'] = df['text'].str.strip().str.replace(r'\s+', ' ', regex=True)

# STEP 2: Encode labels
df['label'] = pd.Categorical(df['job_category']).codes

# STEP 3: Use sklearn to split into train/test
df_train, df_test = train_test_split(df[['text', 'label']], test_size=0.2, random_state=34, stratify=df['label'])

# STEP 4: Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(df_train)
test_dataset = Dataset.from_pandas(df_test)

# STEP 5: Sample 80 training and 20 testing examples
# We are choosing a smaller sample because our program crashes, in the colab, we used the entire dataset
train_df_sampled = df_train.sample(n=80, random_state=42).reset_index(drop=True)
test_df_sampled = df_test.sample(n=20, random_state=42).reset_index(drop=True)

# STEP 6: Convert sampled DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df_sampled)
test_dataset = Dataset.from_pandas(test_df_sampled)

print(f"Training set size: {len(train_dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_train = train_dataset.map(preprocess_function)
tokenized_test = test_dataset.map(preprocess_function)

# Padding and data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Model
num_labels = len(df['label'].unique())
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

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train and evaluate
trainer.train()
evaluation_results = trainer.evaluate()
evaluation_results
print(evaluation_results)