import pandas as pd
import numpy as np
import re
import time
from mistralai import Mistral
from sklearn.metrics import classification_report
from data_cleaning import ResumeDataProcessor
from scoring_metrics import scoring_metrics

# Initialize Mistral client
client = Mistral(api_key="mistral_api_key")  # Replace with your actual key
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

# Few-shot intro
few_shot_intro = (
    "Based on the following resume details, predict the job category from the following options:\n"
    "0: Business Management\n"
    "1: Civil and Mechanical Engineering\n"
    "2: HR & Marketing\n"
    "3: Software & IT\n"
    "4: Tech Engineering\n"
    "Answer with only the integer corresponding to the category."
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
    # Combine all fields
    resume_details = (
        f"Skills: {row.get('skills', '')}\n"
        f"{row.get('degree_names', '')}, {row.get('passing_years', '')}, {row.get('major_field_of_studies', '')}\n"
        f"Positions: {row.get('positions', '')}\n"
        f"Certifications: {row.get('certification_providers', '')} - {row.get('certification_skills', '')}\n"
    )

    # Build the prompt
    prompt = few_shot_intro + "\nResume:\n" + resume_details + "\nAnswer:"

    # API message format
    MESSAGES = [{"role": "user", "content": prompt}]

    # Make API call
    completion = client.chat.complete(
        model=MODEL,
        messages=MESSAGES
    )

    # Get response
    response = completion.choices[0].message.content.strip()
    print(f"\nPrompt:\n{prompt}\nResponse:\n{response}")

    # Extract only the number (0-4) from response
    match = re.search(r"\b([0-4])\b", response)
    if match:
        predictions.append(int(match.group(1)))
    else:
        print(f"Warning: Unexpected format at row {index}")
        predictions.append("ERROR")

    time.sleep(5)

# Turn the predictions into integers
npredictions = [int(x) for x in predictions]

# print a classification report
print(classification_report(resume_sampled_df["label"], npredictions))
