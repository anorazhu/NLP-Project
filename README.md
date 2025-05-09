# Resume Classification Using NLP

## Project Overview

This project builds a set of models to classify candidate resumes into job categories using Natural Language Processing (NLP). By leveraging structured resume fields (skills, degrees, certifications, experience), we implement several classification pipelines ranging from traditional statistical models to modern neural architectures and few-shot prompting using large language models.

---

## Dataset

**Source**: [Kaggle - AI-Powered Resume Screening Dataset](https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset)

**Key Features Used**:
- Skills  
- Degree Names  
- Major Fields of Study  
- Positions  
- Certification Providers & Skills  

**Target Variable**: `job_category`  
Example classes:
- Software & IT  
- Tech Engineering  
- Business Management  
- Civil and Mechanical Engineering  
- HR & Marketing  

---

## Preprocessing

Implemented via `ResumeDataProcessor`:
- Cleans list-like fields
- Flattens structured columns into strings
- Applies `CountVectorizer` to each feature column
- Encodes `job_category` using `LabelEncoder`
- Splits into train/test sets

---

## Models Implemented

### 1. Bag-of-Words (BoW) + Traditional Models
- `naivebayes.py`: Multinomial Naive Bayes  
- `statistical_classification.py`: Logistic Regression and Linear SVC  
- `baseline.py`: Dummy classifiers (most frequent, stratified, uniform) for benchmarking  
- All models evaluated using cross-validation with metrics from `scoring_metrics.py`

> CountVectorizer was used to transform each resume feature into a BoW representation.

---

### 2. Word2Vec Embedding + Deep Learning (`word2vec_representation_classification.py`)
- Pre-trained Word2Vec (Google News 300)
- Models:
  - Gaussian Naive Bayes
  - Support Vector Machine
  - Feedforward Neural Network (FFNN)
  - Convolutional Neural Network (CNN)
- Input: average or padded Word2Vec embeddings
- Balanced training using class weights
- Evaluation via accuracy, precision, recall, F1-score

---

### 3. Transformer-Based Classification with DistilBERT (`bert.py`)
- Combines structured resume fields into a single `text` field
- Uses `distilbert-base-uncased` from Hugging Face for multiclass classification
- Converts data to Hugging Face `Dataset` format
- Tokenizes input and pads sequences with `DataCollatorWithPadding`
- Trained on a sampled subset due to Colab limitations (80 train / 20 test)
- Evaluated using Hugging Face’s `Trainer` with `accuracy` and `F1` metrics

---

### 4. Few-Shot Prompting with Mistral (`mistral_classifier_few_shot.py`)
- Constructs structured prompts from resume fields
- Uses `mistral-large-latest` model to predict job category
- Includes examples in the prompt (few-shot learning)
- Predictions are parsed and evaluated against true labels

---

## Dependencies

All requirements are listed in `requirements.txt`.  
Install with:

```bash
pip install -r requirements.txt
```

---

## File Structure

```
.
├── baseline.py
├── bert.py
├── data.csv
├── data_cleaning.py
├── mistral_classifier_few_shot.py
├── naivebayes.py
├── scoring_metrics.py
├── statistical_classification.py
├── word2vec_representation_classification.py
├── requirements.txt
└── README.md
```

---

## Evaluation

All models are evaluated using:
- Accuracy  
- Precision (macro average)  
- Recall (macro average)  
- F1 Score (macro average)

---

## Future Work

- Hyperparameter tuning for neural networks  
- More granular job category mapping  
- Resume parsing from raw PDFs  
- Zero-shot evaluation with newer LLMs
