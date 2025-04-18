import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset and perform cleaning
class ResumeDataProcessor:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.vectorizers = {}
        self.label_encoder = LabelEncoder()

    def clean_list_fields(self):
        def clean_list_field(cell):
            try:
                if isinstance(cell, str) and cell.startswith('['):
                    parsed = ast.literal_eval(cell)
                    if isinstance(parsed, list):
                        cleaned = [item for item in parsed if str(item).strip().lower() not in ['n/a', 'none', 'nan']]
                        return cleaned if cleaned else np.nan
                return cell
            except Exception:
                return np.nan

        for col in self.df.columns:
            self.df[col] = self.df[col].apply(clean_list_field)

    def flatten_list_fields(self):
        for col in self.df.columns:
            self.df[col] = self.df[col].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else x)

    def prepare_features_and_labels(self):
        # 1. Select all columns except the last two
        X_df = self.df.iloc[:, :-2].astype(str)

        # 2. Apply CountVectorizer to each column and concatenate
        vectors = []
        for col in X_df.columns:
            vec = CountVectorizer()
            X_col = vec.fit_transform(X_df[col])
            self.vectorizers[col] = vec
            vectors.append(X_col.toarray())

        X = np.concatenate(vectors, axis=1)

        # 3. Encode the target labels
        y = self.label_encoder.fit_transform(self.df['job_category'])

        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X, y, X_train, X_test, y_train, y_test

    def process(self):
        self.clean_list_fields()
        self.flatten_list_fields()
        return self.prepare_features_and_labels()
