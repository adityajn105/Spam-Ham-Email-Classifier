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

count_vect = None # count vectorizer 
tfidf_transformer = None	# tfidf transformer
clf = None	# multinomial classifier


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


@app.route('/', methods = ['POST','GET'])
def predict(name=None):
	if request.method == 'POST':
		data = request.form
		text = data['content']
		X_predict = count_vect.transform(np.array([text]))
		X_predict_tfidf = tfidf_transformer.transform(X_predict)
		prediction = clf.predict(X_predict_tfidf)
		return render_template('index.html',data = { 'status' : True, 'result' : prediction[0] ,'content' : '"'+text+'"' } ) 
	else:
		return render_template('index.html',data = { 'status':False }, name=name) 


if __name__ == '__main__':
    print("Initializing Classifier... Please Wait...")
    #getting emails dataframe
    emails = pd.read_csv('DataSets/index/train.csv',sep=" ",header=None,names=['category','paths'])
    emails['body'] = ''
    for index in range(0,len(emails.paths)):
    	fp = codecs.open("DataSets"+emails.paths[index][2:], "r",encoding='utf-8', errors='ignore')
    	email_text = email.message_from_file(fp)
    	texts = ""
    	if  email_text.is_multipart():
    		for part in email_text.get_payload():
    			if part.get_content_maintype() == 'text':
    				texts += part.get_payload()
    	else:
    		texts += email_text.get_payload()
    	texts = re.sub('\n','',texts)
    	texts = re.sub('_','',texts)
    	texts = re.sub('-',' ',texts)
    	texts = re.sub('/',' ',texts)
    	texts = re.sub(':',' ',texts)
    	texts = re.sub('$','',texts)
    	texts = re.sub('=','',texts)
    	texts = re.sub('<.*?>', '', texts)
    	emails.body[index] = " ".join((BeautifulSoup(texts.lower(), 'html.parser').findAll(text=True)))
    	fp.close()
    #removing samples with empty training body    
    temails = emails[emails.body!='']
    temails=temails[['category','body']]

    #making spam and ham samples equal
    te_spam=temails.loc[temails['category']=='spam'].sample(30000,replace=True)
    te_ham=temails.loc[temails['category']=='ham'].sample(30000,replace=True)
    temails=pd.concat([te_spam,te_ham])
    temails.groupby('category').size()

    count_vect = StemmedCountVectorizer(stop_words='english').fit(temails.body)
    X_train_counts = count_vect.transform(temails.body)
    
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)

    clf = MultinomialNB().fit(X_train_tfidf, temails.category)
    print("Classifier Initialized with Accuracy {}".format(clf.score(X_train_tfidf, temails.category)))
    app.run(host = '0.0.0.0', port = 3000)