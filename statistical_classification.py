import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_validate

# Load the dataset
df = pd.read_csv("data.csv")

# Drop rows without a decision
df = df.dropna(subset=['Recruiter Decision'])

# Encode labels: 1 = Hire, 0 = Reject
df['label'] = df['Recruiter Decision'].apply(lambda x: 1 if str(x).strip().lower() == 'hire' else 0)
y = df['label']

# Select and fill text-based columns for bag-of-words
selected_columns = ["Skills", "Experience (Years)", "Education", "Certifications", "Job Role", "Projects Count"]
df[selected_columns] = df[selected_columns].fillna('')

# Bag-of-Words Vectorization
vectorizers = {}
vectors = []
for col in selected_columns:
    vec = CountVectorizer()
    X_vec = vec.fit_transform(df[col].astype(str))
    vectorizers[col] = vec
    vectors.append(X_vec.toarray())

# Combine all vectors into one feature matrix
X_bow_combined = np.concatenate(vectors, axis=1)

# Define evaluation metrics
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

# Logistic Regression with class_weight balanced
print("\n--- Logistic Regression (Balanced) ---")
log_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
log_scores = cross_validate(log_clf, X_bow_combined, y, cv=5, scoring=scoring_metrics)
for metric in scoring_metrics:
    print(f"{metric}: {log_scores['test_' + metric].mean():.4f}")

# Linear SVC with class_weight balanced
print("\n--- Linear SVC (Balanced) ---")
svc_clf = LinearSVC(class_weight="balanced", max_iter=10000)
svc_scores = cross_validate(svc_clf, X_bow_combined, y, cv=5, scoring=scoring_metrics)
for metric in scoring_metrics:
    print(f"{metric}: {svc_scores['test_' + metric].mean():.4f}")
