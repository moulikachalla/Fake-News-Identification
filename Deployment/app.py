import pandas as pd                          # Data Processing Libraries
import numpy as np
import matplotlib.pyplot as plt

import nltk  #(Natural Language Tool Kit - NLTK)                               # NLP Libraries
import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud  # Libaray for Visualization

import os

filepath = "C:/Users/SP/Desktop/Mouli/Deployment/static/images/wc.png"

if os.path.exists(filepath):
    os.remove(filepath)


# Preprocessing functions of Text Data

stop_words = set(stopwords.words('english')) # Stopwords function
wn = WordNetLemmatizer()        # lemmatization

wc = WordCloud(background_color="white")


from flask import Flask, request, jsonify, render_template
from flask import Markup
import joblib
import webbrowser

model = joblib.load('finalmodel.pkl')
vector = joblib.load('vectorizer.pkl')

app = Flask(__name__) #Initialize the flask App

@app.route('/')
def home():
    return render_template('NewsClass.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    article = request.form['newstext']
    # split into words
    tokens = word_tokenize(article)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    words = [w for w in words if not w in stop_words]
    # Lemmatization of words (Root Word Retrieval)
    lemwords = [wn.lemmatize(word) for word in words]
    # Removing Duplicated Words
    impwords = set(lemwords)
    # Joining the words
    impwords = ' '.join(impwords)

    article_tfidf = vector.transform([impwords]).toarray()
    prediction = np.where(model.predict(article_tfidf) == 0, 'FAKE', 'REAL')
    
    # Visualization
    news_text = "".join(impwords)
    wc_img = wc.generate(news_text)
    plt.imshow(wc_img)
    plt.savefig('static/images/wc.png')
    
    return render_template('NewsClass.html', Article_Text = article, prediction = prediction)

if __name__ == "__main__":
    app.run(host='127.0.0.1',port=5000, debug = True)