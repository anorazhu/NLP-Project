from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from data_cleaning import ResumeDataProcessor
from scoring_metrics import scoring_metrics

# Load and process the dataset
processor = ResumeDataProcessor("data.csv")
X, y, X_train, X_test, y_train, y_test, df = processor.process()

# Run dummy classifiers
for strategy in ["most_frequent", "stratified", "uniform"]:
    print(f"\n--- Strategy: {strategy} ---")
    dummy = DummyClassifier(strategy=strategy, random_state=42)
    scores = cross_validate(dummy, X, y, cv=5, scoring=scoring_metrics)

    for metric in scoring_metrics:
        print(f"{metric}: {scores['test_' + metric].mean():.4f}")
