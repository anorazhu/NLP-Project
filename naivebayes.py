from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from data_cleaning import ResumeDataProcessor
from scoring_metrics import scoring_metrics

# Load and process the dataset
processor = ResumeDataProcessor("data.csv")  # Update with your actual file path
X, y, X_train, X_test, y_train, y_test, df = processor.process()

# Train Naive Bayes classifier
mnb = MultinomialNB()
scores = cross_validate(mnb, X, y, cv=5, scoring=scoring_metrics, return_train_score=False)

# Print individual fold results
print("\nMultinomial Naive Bayes (Bag-of-Words) Performance (Each Fold):")
for metric in scoring_metrics:
    print(f"{metric}: {scores['test_' + metric]}")

# Print average scores
print("\nMultinomial Naive Bayes (Bag-of-Words) Average Performance:")
for metric in scoring_metrics:
    mean_score = scores['test_' + metric].mean()
    print(f"{metric}: {mean_score:.4f}")
