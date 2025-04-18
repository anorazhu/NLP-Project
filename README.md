# NLP Project: Resume Classification by Job Category

## Project Goal

The goal of this project is to develop an NLP-based classifier that predicts the most appropriate job category for a candidate based on structured fields from their resume. Rather than working with raw resume text, this project leverages structured data extracted from resumes — such as skills, education, past job titles, and certifications — to perform multiclass classification.

---

## Dataset

We are using this dataset from Kaggle: (https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset). This dataset contains:
- Career Objective
- Skills
- Degree Names
- Institutions
- Major Fields of Study
- Job Experience (Companies, Dates, Positions)
- Certifications
- Responsibilities
and more but these are the features that we will be using for our project. 

There are over 30 job position names, we decided to create a new variable job_category as the target variable

**Target Variable**: `job_category`
**Sample Categories**:
- Software & IT
- Tech Engineering
- Business Management
- Civil and Mechanical Engineering
- HR & Marketing
