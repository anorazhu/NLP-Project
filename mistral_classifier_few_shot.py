import pandas as pd
import numpy as np
import re
import time
from mistralai import Mistral
from sklearn.metrics import classification_report
from data_cleaning import ResumeDataProcessor
from scoring_metrics import scoring_metrics

# Initialize Mistral client
client = Mistral(api_key="w6yh0tpXA6gK3p52Yt1vfzM8Yv74LkNX")  # Replace with your actual key
MODEL = "mistral-large-latest"

# Load and process data
processor = ResumeDataProcessor("data.csv")
_, _, _, _, _, _, df = processor.process()

# Create text column from all features (drop NaNs and join as one string)
X_df = df.iloc[:, :-2]
df['text'] = X_df.apply(lambda row: ' '.join([str(val) for val in row if pd.notna(val) and str(val).lower() != 'nan']), axis=1)
df['text'] = df['text'].str.strip().str.replace(r'\s+', ' ', regex=True)

# Encode labels
df['label'] = pd.Categorical(df['job_category']).codes

# Sample 100 random resumes
resume_sampled_df = df.sample(n=100, random_state=42).reset_index(drop=True)
y_sampled = resume_sampled_df["label"].tolist()

# Few-shot intro
few_shot_intro = (
    "Based on the following resume details, predict the job category from the following options:\n"
    "0: Business Management\n"
    "1: Civil and Mechanical Engineering\n"
    "2: HR & Marketing\n"
    "3: Software & IT\n"
    "4: Tech Engineering\n"
    "Answer with only the integer corresponding to the category.\n\n"
    "Example 1:\n"
    "Skills: Python, JavaScript, SQL, REST APIs\n"
    "Degree: B.S. in Computer Science\n"
    "Positions: Full Stack Developer\n"
    "Certifications: AWS Developer\n"
    "Answer: 3\n\n"
    "Example 2:\n"
    "Skills: Machine Learning, PyTorch, TensorFlow, AWS\n"
    "Degree: M.S. in Data Science\n"
    "Positions: ML Engineer\n"
    "Certifications: TensorFlow Certified\n"
    "Answer: 4\n\n"
    "Example 3:\n"
    "Skills: Budget Planning, Forecasting\n"
    "Degree: MBA in Finance\n"
    "Positions: Analyst\n"
    "Certifications: CMA\n"
    "Answer: 0\n"
)

# Send prompts to Mistral
predictions = []
for index, row in resume_sampled_df.iterrows():
    resume_details = (
        f"Skills: {row.get('skills', '')}\n"
        f"Degree: {row.get('degree_names', '')}, {row.get('passing_years', '')}, {row.get('major_field_of_studies', '')}\n"
        f"Positions: {row.get('positions', '')}\n"
        f"Certifications: {row.get('certification_providers', '')} - {row.get('certification_skills', '')}\n"
    )
    prompt = few_shot_intro + "\nResume:\n" + resume_details + "\nAnswer:"
    MESSAGES = [{"role": "user", "content": prompt}]
    
    try:
        completion = client.chat.complete(model=MODEL, messages=MESSAGES)
        response = completion.choices[0].message.content.strip()
        match = re.search(r"\b([0-4])\b", response)
        predictions.append(int(match.group(1)) if match else "ERROR")
    except Exception as e:
        predictions.append("ERROR")
    
    time.sleep(5)

# Filter predictions and compute evaluation
valid_preds = [p for p in predictions if p != "ERROR"]
valid_labels = [y_sampled[i] for i, p in enumerate(predictions) if p != "ERROR"]

# Print classification report
report = classification_report(valid_labels, valid_preds, output_dict=False)
print(report)
