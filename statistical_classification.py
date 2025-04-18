from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from data_cleaning import ResumeDataProcessor
from scoring_metrics import scoring_metrics

# Load and process the dataset
processor = ResumeDataProcessor("data.csv")
X, y, _, _, _, _ = processor.process()

# --- Logistic Regression ---
print("\n--- Logistic Regression ---")
log_reg = LogisticRegression(max_iter=1000)
log_scores = cross_validate(log_reg, X, y, cv=5, scoring=scoring_metrics)

for metric in scoring_metrics:
    print(f"{metric}: {log_scores['test_' + metric].mean():.4f}")

# --- Linear SVC ---
print("\n--- Linear SVC ---")
svc = LinearSVC(max_iter=10000)
svc_scores = cross_validate(svc, X, y, cv=5, scoring=scoring_metrics)

for metric in scoring_metrics:
    print(f"{metric}: {svc_scores['test_' + metric].mean():.4f}")