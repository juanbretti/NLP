# %%
# Load the libraries
import nltk
from nltk.stem import *
from nltk.corpus import stopwords
nltk.download('stopwords')

import re
import string

import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

import joblib

from helpers import helpers

# %%
## Load and transform data ----
train_df = helpers.fetch('train')
test_df = helpers.fetch('test')

train_df = helpers.multiple_filters(train_df)
test_df = helpers.multiple_filters(test_df)

train_df = helpers.process_text_additional(train_df)
test_df = helpers.process_text_additional(test_df)

train_df, encoder_le, target_labels = helpers.f_encoder_le(train_df)
test_df, _, _ = helpers.f_encoder_le(test_df, encoder_le)

train_df_tfidf, encoder_cv, encoder_tfidf = helpers.f_encoder_cv_tfidf(train_df)
test_df_tfidf, _, _ = helpers.f_encoder_cv_tfidf(test_df, encoder_cv, encoder_tfidf)

# %%
# Train the model
def train_and_evaluate_classifier(X, yt, estimator, grid):
    """Train and Evaluate a estimator (defined as input parameter) on the given labeled data using accuracy."""
    
    # Cross validation
    from sklearn.model_selection import ShuffleSplit
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
      
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=estimator, cv=cv,  param_grid=grid, error_score=0.0, n_jobs = -1, verbose = 0)
    
    # Train the model over and tune the parameters
    print("Training model")
    grid_search.fit(X, yt)

    # CV-score
    print("Best score: %0.3f" % grid_search.best_score_)
    if len(grid) > 0:
        best_parameters = grid_search.best_estimator_.get_params()
        print("Best parameters set:",best_parameters)
    return grid_search, best_parameters

# Space of hyperparameter optimization
svm_grid = [
  {'C': [0.01, 0.1, 1], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']},
 ]
svm_clf = SVC(probability=True, random_state=42)

SVM_gridsearch, SVM_best_params = train_and_evaluate_classifier(train_df_tfidf, train_df['class_group_encoded'], svm_clf, svm_grid)

# %%
# Prediction
y_predict = SVM_gridsearch.predict(test_df_tfidf)
y_predict_proba = SVM_gridsearch.predict_proba(test_df_tfidf)

# Performance of the model
print(classification_report(test_df['class_group_encoded'], y_predict))

# %%
# Dump the model to the local storage
joblib.dump(SVM_gridsearch, './models/SVM_gridsearch.pkl')
joblib.dump(encoder_le, './models/encoder_le.pkl')
joblib.dump(target_labels, './models/target_labels.pkl')
joblib.dump(encoder_cv, './models/encoder_cv.pkl')
joblib.dump(encoder_tfidf, './models/encoder_tfidf.pkl')
joblib.dump(test_df, './models/test_df.pkl')
joblib.dump(test_df_tfidf, './models/test_df_tfidf.pkl')
joblib.dump(y_predict, './models/y_predict.pkl')
joblib.dump(y_predict_proba, './models/y_predict_proba.pkl')

# Load the model from the local storage
# SVM_gridsearch = joblib.load('./models/SVM_gridsearch.pkl')
# encoder_le = joblib.load('./models/encoder_le.pkl')
# target_labels = joblib.load('./models/target_labels.pkl')
# encoder_cv = joblib.load('./models/encoder_cv.pkl')
# encoder_tfidf = joblib.load('./models/encoder_tfidf.pkl')
# test_df = joblib.load('./models/test_df.pkl')
# test_df_tfidf = joblib.load('./models/test_df_tfidf.pkl')
# y_predict = joblib.load('./models/y_predict.pkl')
# y_predict_proba = joblib.load('./models/y_predict_proba.pkl')

# %%