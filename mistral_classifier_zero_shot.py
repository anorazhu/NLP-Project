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

# 1. Initialize Processor and load data
processor = ResumeDataProcessor("data.csv")
X, y, X_train, X_test, y_train, y_test, df = processor.process()
df['label'] = pd.Categorical(df['job_category']).codes

# 2. Instead of X_test.index, directly split the df (correct)
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 3. Sample from df_test (the real original resumes)
resume_sampled_df = df_test.sample(n=100, random_state=42).reset_index(drop=True)

# 4. Get true labels
y_sampled = resume_sampled_df["label"].tolist()

# 5. Build prompts and send to Mistral
predictions = []

for index, row in resume_sampled_df.iterrows():
    prompt = (
        "Based on the following resume details, predict the job category from the following options:\n"
        "0: Business Management\n"
        "1: Civil and Mechanical Engineering\n"
        "2: HR & Marketing\n"
        "3: Software & IT\n"
        "4: Tech Engineering\n"
        f"Resume:\n"
        f"Skills: {row.get('skills', '')}\n"
        f"Degree: {row.get('degree_names', '')}, {row.get('passing_years', '')}, {row.get('major_field_of_studies', '')}\n"
        f"Positions: {row.get('positions', '')}\n"
        f"Certifications: {row.get('certification_providers', '')} - {row.get('certification_skills', '')}\n"
        "Answer with only the integer corresponding to the category."
    )

    MESSAGES = [{"role": "user", "content": prompt}]
    
    completion = client.chat.complete(
        model=MODEL,
        messages=MESSAGES
    )
    
    response = completion.choices[0].message.content.strip()
    match = re.search(r"\b([0-4])\b", response)
    predictions.append(int(match.group(1)) if match else "ERROR")

    time.sleep(5)

# 6. Clean predictions (remove ERRORs if any)
npredictions = [int(x) for i, x in enumerate(predictions) if x != "ERROR"]
y_sampled_cleaned = [y_sampled[i] for i, x in enumerate(predictions) if x != "ERROR"]

# 7. Report
print(classification_report(npredictions, y_sampled_cleaned))
