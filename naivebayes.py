import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Load the dataset
df = pd.read_csv("data.csv")

# Encode label: "Hire" = 1, "Reject" = 0
df = df[df['Recruiter Decision'].notna()]
df['label'] = df['Recruiter Decision'].map({'Hire': 1, 'Reject': 0})

# Select the relevant text columns
selected_columns = ['Skills', 'Education', 'Certifications', 'Job Role']

# Initialize list to hold vectorized feature arrays
vectorizers = {}
vectors = []

# Apply CountVectorizer separately on each text column
for col in selected_columns:
    vectorizer = CountVectorizer()
    X_col = vectorizer.fit_transform(df[col].astype(str))  # Ensure column is string
    vectorizers[col] = vectorizer
    vectors.append(X_col.toarray())

# Combine all vectorized features horizontally
X_combined = np.concatenate(vectors, axis=1)
y = df['label'].values

# Define scoring metrics
scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']

# Train and evaluate Multinomial Naive Bayes using 5-fold cross-validation
mnb = MultinomialNB()
scores = cross_validate(mnb, X_combined, y, cv=5, scoring=scoring_metrics)

# Print the results
print("\nMultinomial Naive Bayes (Bag-of-Words) Performance:")
for metric in scoring_metrics:
    print(f"{metric}: {scores['test_' + metric].mean():.4f}")
