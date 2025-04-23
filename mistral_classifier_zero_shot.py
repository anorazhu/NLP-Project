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
y_sampled = resume_sampled_df["label"].tolist()

# Store predictions
predictions = []

# Loop through the sampled resumes
for index, row in resume_sampled_df.iterrows():
    # Create the prompt using the text field from the current row
    prompt = (
        "Based on the following resume details, predict the job category from the following options:\n"
        "0: Business Management\n"
        "1: Civil and Mechanical Engineering\n"
        "2: HR & Marketing\n"
        "3: Software & IT\n"
        "4: Tech Engineering\n"
        f"Resume:\n{row['text']}\n"
        "Answer with only the integer corresponding to the category."
    )

    # Put it in MESSAGES format for Mistral
    MESSAGES = [{"role": "user", "content": prompt}]


    # Make the LLM request
    completion = client.chat.complete(
        model=MODEL,
        messages=MESSAGES
    )
    response = completion.choices[0].message.content.strip()

    # Print prompt and response
    print(f"Prompt:\n{prompt}")
    print(f"Prediction:\n{completion.choices[0].message.content}")

    # Save the response
    match = re.search(r"\b([0-4])\b", response)
    predictions.append(int(match.group(1)) if match else "ERROR")

    # Pause to avoid hitting Mistral rate limits
    time.sleep(5)


# Turn the predictions into integers
npredictions = [int(x) for x in predictions]

# print a classification report
print(classification_report(npredictions, resume_sampled_df["label"]))