from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
import pandas as pd

# Load your dataset
df = pd.read_csv("data.csv") 

# Define features and labels
X = df.drop(columns=['Recruiter Decision'])  # Use all columns except target
y = df['Recruiter Decision']  # This is the label you're predicting

# Define evaluation metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Try multiple dummy classifier strategies
for strategy in ['most_frequent', 'stratified', 'uniform']:
    print(f"\n--- Baseline Strategy: {strategy} ---")
    
    dummy_clf = DummyClassifier(strategy=strategy, random_state=42)
    
    # Run 5-fold cross-validation
    scores = cross_validate(dummy_clf, X, y, cv=5, scoring=scoring)
    
    # Print average of each metric
    for metric in scoring:
        mean_score = scores[f'test_{metric}'].mean()
        print(f"{metric}: {mean_score:.4f}")
