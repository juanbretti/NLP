# %%
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

# %%
## Load data ----
# Train
dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
train_df = pd.DataFrame()
train_df['text'] = dataset.data
train_df['source'] = dataset.target
train_df['class'] = [dataset.target_names[i] for i in train_df['source']]

# Test
dataset = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
test_df = pd.DataFrame()
test_df['text'] = dataset.data
test_df['source'] = dataset.target
test_df['class'] = [dataset.target_names[i] for i in test_df['source']]

# %%
class_conversion_dict = {'talk.politics.misc':'politics',
    'talk.politics.guns':'politics',
    'talk.politics.mideast':'politics',
    'rec.sport.hockey':'sport',
    'rec.sport.baseball':'sport',
    'soc.religion.christian':'religion',
    'talk.religion.misc':'religion',
    'alt.atheism':'religion',
    'comp.windows.x':'computer',
    'comp.sys.ibm.pc.hardware':'computer',
    'comp.os.ms-windows.misc':'computer',
    'comp.graphics':'computer',
    'comp.sys.mac.hardware':'computer',
    'misc.forsale':'sales',
    'rec.autos':'automobile',
    'rec.motorcycles':'automobile',
    'sci.crypt':'science',
    'sci.electronics':'science',
    'sci.space':'science',
    'sci.med':'medicine'}
df_class_conversion_dict = pd.DataFrame(list(class_conversion_dict.items()),columns = ['class','class_group']) 

train_df = train_df.merge(df_class_conversion_dict, on='class')
test_df = test_df.merge(df_class_conversion_dict, on='class')

# %%
def multiple_filters(df):
    # Classes of interest
    df = df[df['class_group'].isin(['religion', 'automobile', 'medicine', 'sport'] )]
    # No text
    df['words_count'] = df['text'].apply(lambda x:len(str(x).split()))
    no_text = df[df['words_count']==0]
    df = df.drop(no_text.index)
    return df

train_df = multiple_filters(train_df)
test_df = multiple_filters(test_df)

# %%
def clean(email):            
    # Special characters
    email = re.sub(r"\x89Û_", "", email)
    email = re.sub(r"\x89ÛÒ", "", email)
    email = re.sub(r"\x89ÛÓ", "", email)
    email = re.sub(r"\x89ÛÏWhen", "When", email)
    email = re.sub(r"\x89ÛÏ", "", email)
    email = re.sub(r"China\x89Ûªs", "China's", email)
    email = re.sub(r"let\x89Ûªs", "let's", email)
    email = re.sub(r"\x89Û÷", "", email)
    email = re.sub(r"\x89Ûª", "", email)
    email = re.sub(r"\x89Û\x9d", "", email)
    email = re.sub(r"å_", "", email)
    email = re.sub(r"\x89Û¢", "", email)
    email = re.sub(r"\x89Û¢åÊ", "", email)
    email = re.sub(r"fromåÊwounds", "from wounds", email)
    email = re.sub(r"åÊ", "", email)
    email = re.sub(r"åÈ", "", email)
    email = re.sub(r"JapÌ_n", "Japan", email)    
    email = re.sub(r"Ì©", "e", email)
    email = re.sub(r"å¨", "", email)
    email = re.sub(r"SuruÌ¤", "Suruc", email)
    email = re.sub(r"åÇ", "", email)
    email = re.sub(r"å£3million", "3 million", email)
    email = re.sub(r"åÀ", "", email)
            
    # Character entity references
    email = re.sub(r"&gt;", ">", email)
    email = re.sub(r"&lt;", "<", email)
    email = re.sub(r"&amp;", "&", email)
    
    # Typos, slang and informal abbreviations
    email = re.sub(r"w/e", "whatever", email)
    email = re.sub(r"w/", "with", email)
    email = re.sub(r"USAgov", "USA government", email)
    email = re.sub(r"recentlu", "recently", email)
    email = re.sub(r"Ph0tos", "Photos", email)
    email = re.sub(r"amirite", "am I right", email)
    email = re.sub(r"exp0sed", "exposed", email)
    email = re.sub(r"<3", "love", email)
    email = re.sub(r"amageddon", "armageddon", email)
    email = re.sub(r"Trfc", "Traffic", email)
    email = re.sub(r"8/5/2015", "2015-08-05", email)
    email = re.sub(r"WindStorm", "Wind Storm", email)
    email = re.sub(r"8/6/2015", "2015-08-06", email)
    email = re.sub(r"10:38PM", "10:38 PM", email)
    email = re.sub(r"10:30pm", "10:30 PM", email)
    email = re.sub(r"16yr", "16 year", email)
    email = re.sub(r"lmao", "laughing my ass off", email)   
    email = re.sub(r"TRAUMATISED", "traumatized", email)
        
    # Words with punctuations and special characters
    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"
    for p in punctuations:
        email = email.replace(p, f' {p} ')
        
    return email

def process_text_additional(raw_text):
    raw_text = clean(raw_text)

    letters_only = re.sub("[^a-zA-Z]", " ",raw_text) 
    letters_only = letters_only.lower()
    letters_only = re.sub('\[.*?\]', '', letters_only)
    letters_only= re.sub('http?://\S+|www\.\S+', '', letters_only)
    letters_only = re.sub('<.*?>+', '', letters_only)
    letters_only = re.sub('[%s]' % re.escape(string.punctuation), '', letters_only)
    letters_only = re.sub('\n', '', letters_only)
    letters_only = re.sub('\w*\d\w*', '', letters_only)
    words = letters_only.lower().split()
    
    stops = set(stopwords.words("english"))  
    not_stop_words = [w for w in words if not w in stops]
    
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(word) for word in not_stop_words]
    
    return( " ".join( stemmed ))

train_df['clean_text'] = train_df['text'].apply(lambda x: process_text_additional(x))
test_df['clean_text'] = test_df['text'].apply(lambda x: process_text_additional(x))

# %%
# TODO: Necesito `le`
encoder_le = preprocessing.LabelEncoder()
encoder_le.fit(train_df['class_group'])
target_labels = encoder_le.classes_
train_df['class_group_encoded'] = encoder_le.transform(train_df['class_group'])
test_df['class_group_encoded'] = encoder_le.transform(test_df['class_group'])

# %%
# TODO: Necesito `count_vect` y `tfidf`
encoder_cv = CountVectorizer(analyzer = "word")
encoder_cv.fit(train_df['clean_text'])
train_df_clean_text_cv = encoder_cv.transform(train_df['clean_text'])
test_df_clean_text_cv = encoder_cv.transform(test_df['clean_text'])

encoder_tfidf = TfidfTransformer(norm="l2")
encoder_tfidf.fit(train_df_clean_text_cv)
train_df_clean_text_tfidf = encoder_tfidf.transform(train_df_clean_text_cv)
test_df_clean_text_tfidf = encoder_tfidf.transform(test_df_clean_text_cv)

# %%
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

svm_grid = [
  {'C': [0.01, 0.1, 1], 'kernel': ['linear']},
#   {'C': [1, 10, 100, 1000], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']},
 ]
svm_clf = SVC(probability=True, random_state=42)

SVM_gridsearch, SVM_best_params = train_and_evaluate_classifier(train_df_clean_text_tfidf, train_df['class_group_encoded'], svm_clf, svm_grid)

# %%
# joblib.dump(SVM_gridsearch, './models/SVM_gridsearch.pkl')
# SVM_gridsearch = joblib.load('./models/SVM_gridsearch.pkl')

y_predict = SVM_gridsearch.predict(test_df_clean_text_tfidf)
y_predict_proba = SVM_gridsearch.predict_proba(test_df_clean_text_tfidf)

print(classification_report(test_df['class_group_encoded'], y_predict))

# %%