import numpy as np
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler
from gensim.utils import simple_preprocess

from data_cleaning import ResumeDataProcessor
from scoring_metrics import scoring_metrics

# Load and preprocess resume data
processor = ResumeDataProcessor("data.csv")
_, y, _, _, _, _, df = processor.process()

# Generate 'text' and 'tokens' columns if not present
if 'text' not in df.columns:
    X_df = df.drop(columns=["job_position_name", "job_category", "label"], errors="ignore").copy()
    df['text'] = X_df.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

df['tokens'] = df['text'].apply(simple_preprocess)

# Load pretrained Word2Vec model
bigmodel = gensim.models.KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300-SLIM.bin.gz", binary=True
)

# Average Word2Vec embeddings per resume
def get_avg_w2v(text):
    tokens = str(text).split()
    vectors = [bigmodel[word] for word in tokens if word in bigmodel]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

X_vectors = np.vstack(df['tokens'].apply(lambda tokens: get_avg_w2v(" ".join(tokens))).values)
X_scaled = StandardScaler().fit_transform(X_vectors)

# --- Logistic Regression ---
clf_log = LogisticRegression(max_iter=1000)
scores_log = cross_validate(clf_log, X_scaled, y, cv=5, scoring=scoring_metrics)
print("--- Logistic Regression ---")
for metric in scoring_metrics:
    print(f"{metric}: {scores_log['test_' + metric].mean():.4f}")

# --- Gaussian Naive Bayes ---
clf_nb = GaussianNB()
scores_nb = cross_validate(clf_nb, X_scaled, y, cv=5, scoring=scoring_metrics)
print("\n--- Gaussian Naive Bayes ---")
for metric in scoring_metrics:
    print(f"{metric}: {scores_nb['test_' + metric].mean():.4f}")

# --- SVM (Linear Kernel) ---
clf_svm = SVC(kernel='linear', class_weight='balanced')
scores_svm = cross_validate(clf_svm, X_scaled, y, cv=5, scoring=scoring_metrics)
print("\n--- SVM (linear kernel) ---")
for metric in scoring_metrics:
    print(f"{metric}: {scores_svm['test_' + metric].mean():.4f}")

# --- LinearSVC ---
clf_linsvc = LinearSVC(max_iter=10000, class_weight='balanced')
scores_linsvc = cross_validate(clf_linsvc, X_scaled, y, cv=5, scoring=scoring_metrics)
print("\n--- LinearSVC ---")
for metric in scoring_metrics:
    print(f"{metric}: {scores_linsvc['test_' + metric].mean():.4f}")
