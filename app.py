from flask import Flask,render_template,request,redirect

import pandas as pd
import numpy as np

import email
import codecs
import re
from bs4 import BeautifulSoup

import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

@app.route('/')
def home(name=None):
	return render_template('Booths-Multiplier.html',data = {'status':False}, name=name) 

@app.route('/', methods = ['POST'])
def predict():
	
if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3000)
