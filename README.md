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
  - LinearSVC
  - Logistic Regression
  - Feedforward Neural Network (FFNN)
  - Convolutional Neural Network (CNN)
- Input: average or padded Word2Vec embeddings
- Balanced training using class weights
- Evaluation via accuracy, precision, recall, F1-score

#### Recommendation: It is better to run Word2Vec-based models in the provided Google Colab notebook. We had to downgrade Python to version 3.11 to work with gensim and other required libraries, so using Colab helps avoid potential compatibility issues.
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

## How to Run

### Prerequisites
Ensure you have Python 3.11+ and install dependencies:

```bash
pip install -r requirements.txt
```

### Order to Run
1. Preprocess data  
   Run: `data_cleaning.py`  
   Output: cleaned `df` and encoded features

2. Statistical models (optional)  
   Run: `naivebayes.py`, `statistical_classification.py`, `baseline.py`

3. Word2Vec-based models  
   Run: `word2vec_representation_classification.py` (recommended in Colab)

4. BERT-based model  
   Run: `bert.py`

5. Mistral few-shot prompting  
   Run: `mistral_classifier_few_shot.py` (requires Mistral API key)

---

## File Structure

```
.
├── baseline.py                         # Dummy classifier benchmarks
├── bert.py                             # DistilBERT classification
├── data.csv                            # Resume dataset from Kaggle
├── data_cleaning.py                    # ResumeDataProcessor class
├── mistral_classifier_few_shot.py      # Mistral prompting models
├── naivebayes.py                       # Multinomial NB with BoW
├── scoring_metrics.py                  # Accuracy, Precision, Recall, F1
├── statistical_classification.py       # LogisticRegression, LinearSVC
├── word2vec_representation_classification.py  # W2V + ML/DL models
├── word2vec_statistical_classification.py     # W2V + Logistic/SVM
├── word2vec_colab.ipynb                # Colab version of Word2Vec pipeline
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

- Improve overall classification accuracy across all job categories
- Use BERT-based embeddings with k-NN to capture semantic similarity
- Build a resume-to-job matching system using predicted job categories
- Filter real job listings based on predicted category
- Compare resume and job posting embeddings using cosine similarity
- Rank job listings to recommend the best-fit positions for each resume

---

## External Resources

- **Pretrained Word2Vec**: [Google News Vectors](https://code.google.com/archive/p/word2vec/)  
- **Mistral API**: [Mistral Docs](https://docs.mistral.ai/)  

