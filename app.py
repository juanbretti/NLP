# %%
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from flask_sqlalchemy  import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from wtforms import StringField, PasswordField, BooleanField, TextAreaField
from wtforms.validators import InputRequired, Email, Length
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import joblib
from helpers import helpers

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

class ComposeForm(FlaskForm):
    message = TextAreaField('message', validators=[InputRequired()])

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
                return redirect(url_for('inbox'))

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

def find_class_prob(y_predict_proba):
    target_labels = joblib.load('./models/target_labels.pkl')

    # Top
    n=2
    best_n = np.argsort(-y_predict_proba, axis=1)[:,:n]
    label1, probability1 = [], []
    label2, probability2 = [], []
    for best, prob in zip(best_n, y_predict_proba):
        # Top #1
        label1.append(target_labels[best[0]])
        probability1.append(prob[best[0]])
        # Top #2
        label2.append(target_labels[best[1]])
        probability2.append(prob[best[1]])

    test_class_prob = pd.DataFrame({'Class 1': label1, 'Probability 1': probability1, 'Class 2': label2, 'Probability 2': probability2})
    return test_class_prob

@app.route('/inbox')
@login_required
def inbox():
    test_df = joblib.load('./models/test_df.pkl')
    test_df = test_df.reset_index()
    y_predict_proba = joblib.load('./models/y_predict_proba.pkl')
    test_class_prob = find_class_prob(y_predict_proba)
    test_df_contact = pd.concat([test_df, test_class_prob], axis=1)

    return render_template('inbox.html', name=current_user.username, df=test_df_contact.sample(n=20))

@app.route('/compose', methods=['GET', 'POST'])
@login_required
def compose():
    
    form = ComposeForm()
    predicted_class='Write your email'

    if request.method == 'POST':
        if form.validate_on_submit():
            # Load model and encoders
            SVM_gridsearch = joblib.load('./models/SVM_gridsearch.pkl')
            encoder_le = joblib.load('./models/encoder_le.pkl')
            encoder_cv = joblib.load('./models/encoder_cv.pkl')
            encoder_tfidf = joblib.load('./models/encoder_tfidf.pkl')

            # Apply encoders
            test_df = pd.DataFrame({'text': [form.message.data]})
            test_df = helpers.process_text_additional(test_df)
            test_df_tfidf, _ = helpers.f_encoder_cv(test_df, encoder_cv)
            test_df_tfidf, _ = helpers.f_encoder_tfidf(test_df_tfidf, encoder_tfidf)

            # Transform
            # y_predict = SVM_gridsearch.predict(test_df_tfidf)
            y_predict_proba = SVM_gridsearch.predict_proba(test_df_tfidf)
            test_class_prob = find_class_prob(y_predict_proba)
            test_df_contact = pd.concat([test_df, test_class_prob], axis=1)
            
            form = ComposeForm(message=form.message)
            # <td> {{ email['Class 1'] }} [{{ '%0.0f'|format((email['Probability 1']|float)*100) }}%]</td>
            predicted_class= f"{test_df_contact['Class 1'][0]} [{test_df_contact['Probability 1'][0]*100:0.0f}%] \
                or {test_df_contact['Class 2'][0]} [{test_df_contact['Probability 2'][0]*100:0.0f}%]"
        else:
            form = ComposeForm(message='')
            predicted_class='Write your email'

    return render_template('compose.html', name=current_user.username, form=form, predicted_class=predicted_class)

# %%
if __name__ == '__main__':
    app.run(debug=True)

# %%

