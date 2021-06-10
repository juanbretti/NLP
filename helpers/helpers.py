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
def fetch(subset):
    dataset = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
    train_df = pd.DataFrame()
    train_df['text'] = dataset.data
    train_df['source'] = dataset.target
    train_df['class'] = [dataset.target_names[i] for i in train_df['source']]

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
    df_class_conversion_dict = pd.DataFrame(list(class_conversion_dict.items()),columns = ['class', 'class_group']) 

    train_df = train_df.merge(df_class_conversion_dict, on='class')

    return train_df

def multiple_filters(df):
    # Classes of interest
    df = df[df['class_group'].isin(['religion', 'automobile', 'medicine', 'sport'])]
    # No text
    df['words_count'] = df['text'].apply(lambda x:len(str(x).split()))
    no_text = df[df['words_count']==0]
    df = df.drop(no_text.index)
    return df

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

def process_text_additional_row(raw_text):
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

def process_text_additional(train_df):
    train_df['clean_text'] = train_df['text'].apply(lambda x: process_text_additional_row(x))
    return train_df

def f_encoder_le(train_df, encoder_le=None):
    if encoder_le is None:
        encoder_le = preprocessing.LabelEncoder()
        encoder_le.fit(train_df['class_group'])
    target_labels = encoder_le.classes_
    train_df['class_group_encoded'] = encoder_le.transform(train_df['class_group'])
    return train_df, encoder_le, target_labels

def f_encoder_cv_tfidf(train_df, encoder_cv=None, encoder_tfidf=None):
    if encoder_cv is None:
        encoder_cv = CountVectorizer(analyzer = "word")
        encoder_cv.fit(train_df['clean_text'])
    train_df_clean_text_cv = encoder_cv.transform(train_df['clean_text'])

    if encoder_tfidf is None:
        encoder_tfidf = TfidfTransformer(norm="l2")
        encoder_tfidf.fit(train_df_clean_text_cv)
    train_df_clean_text_tfidf = encoder_tfidf.transform(train_df_clean_text_cv)
    
    return train_df_clean_text_tfidf, encoder_cv, encoder_tfidf

# %%