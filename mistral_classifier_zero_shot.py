import pandas as pd
import numpy as np
import re
import time
from mistralai import Mistral
from sklearn.metrics import classification_report
from data_cleaning import ResumeDataProcessor
from scoring_metrics import scoring_metrics 

# Initialize Mistral client
client = Mistral(api_key="w6yh0tpXA6gK3p52Yt1vfzM8Yv74LkNX")  
MODEL = "mistral-large-latest"

# Load and process data
processor = ResumeDataProcessor("data.csv")
_, _, _, _, _, _, df = processor.process()

# Combine all text features into one "text" column
feature_columns = df.columns[df.columns != 'job_category']
df['text'] = df[feature_columns].apply(
    lambda row: ' '.join(
        str(val).replace('\n', ' ').strip()
        for val in row
        if pd.notna(val) and str(val).strip().lower() != 'nan'
    ),
    axis=1
)

# Encode labels
df['label'] = pd.Categorical(df['job_category']).codes

# Sample 100 resumes 
resume_sampled_df = df.sample(n=100, random_state=42).reset_index(drop=True)

# Store predictions
predictions = []

# Loop through the sampled resumes
for index, row in resume_sampled_df.iterrows():
    # Create the prompt using the clean resume text
    prompt = (
        "Based on the following resume details, predict the job category from the following options: "
        "['Software & IT', 'Tech Engineering', 'Civil and Mechanical Engineering', "
        "'Business Management', 'HR & Marketing'].\n"
        f"Resume:\n{row['text']}\n"
        "Answer with only the job category name."
    )

    MESSAGES = [{"role": "user", "content": prompt}]

    try:
        # Send to Mistral API
        completion = client.chat.complete(
            model=MODEL,
            messages=MESSAGES
        )

        # Extract and store the prediction
        response = completion.choices[0].message.content.strip()
        print(f"\nPrompt:\n{prompt}\nPrediction:\n{response}")
        predictions.append(response)

    except Exception as e:
        print(f"Error at row {index}: {e}")
        predictions.append("ERROR")

    # Wait to avoid hitting rate limits
    time.sleep(5)

# Convert string predictions to label-encoded integers
from sklearn.preprocessing import LabelEncoder

encoded_predictions = processor.label_encoder.transform(predictions)

# True labels from the sampled DataFrame
true_labels = resume_sampled_df['label'].values

# Print classification report
print(classification_report(true_labels, encoded_predictions, target_names=processor.label_encoder.classes_))
