# %%
from flask import Flask, render_template, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from flask_sqlalchemy  import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from werkzeug.security import generate_password_hash, check_password_hash

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
app = Flask(__name__)
app.config['SECRET_KEY'] = 'GroupCSecretKey2021-2022'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./database/database.db'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# %%
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

class LoginForm(FlaskForm):
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])

# %%
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return '<h1>Invalid username or password</h1>'
        #return '<h1>' + form.username.data + ' ' + form.password.data + '</h1>'

    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return '<h1>New user has been created!</h1>'
        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'

    return render_template('signup.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', name=current_user.username)

# %%
# if __name__ == '__main__':
    # app.run(debug=True)

# %%
## Load data ----
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

# {'politics':0, 'sport':1, 'religion':2, 'computer':3, 'sales':4, 'automobile':5, 'science':6, 'medicine':7}

test_df = test_df.merge(df_class_conversion_dict, on='class')

# %%
test_df = test_df[test_df['class_group'].isin(['religion', 'automobile', 'medicine','sport'] )]

def filter_no_text(test_df):
    test_df['words_count'] = test_df['text'].apply(lambda x:len(str(x).split()))
    no_text = test_df[test_df['words_count']==0]
    test_df.drop(no_text.index,inplace=True)
    return test_df

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

test_df['clean_text'] = test_df['text'].apply(lambda x: process_text_additional(x))

# %%
# TODO: Necesito `le`
le = preprocessing.LabelEncoder()
le.fit(dataset.target.values)
target_labels = le.classes_
dataset.target = le.transform(test_df[])

# %%
# TODO: Necesito `count_vect` y `tfidf`
count_vect = CountVectorizer(analyzer = "word")
train_features = count_vect.fit_transform(X_train['clean_text'])

tfidf = TfidfTransformer(norm="l2")
train_text_tfidf_features = tfidf.fit_transform(train_features)

testData = count_vect.transform(X_test['clean_text'])
test_text_tfidf_features = tfidf.transform(testData)

# %%
# joblib.dump(SVM_classifier, './models/SVM_classifier.pkl')
SVM_classifier = joblib.load('./models/SVM_classifier.pkl')

y_predict = SVM_classifier.predict(test_text_tfidf_features)

print(classification_report(y_test, y_predict))