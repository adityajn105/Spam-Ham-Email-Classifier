import pandas as pd
import numpy as np

emails = pd.read_csv('DataSets/index/train.csv',sep=" ",header=None,names=['category','paths'],nrows=10000,skiprows=40000)
emails['body'] = ''
#emails.groupby("category").size()

import email
import codecs
import re
from bs4 import BeautifulSoup
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
#emails.head(10)

temails = emails[emails.body!='']
#temails.head(10)

#applying bag of words algorithm
#segment each text file into words (for English splitting by space), and 
#count # of times each word occurs in each document and finally assign each word an integer id.
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer().fit(temails.body)
X_train_counts = count_vect.transform(temails.body)
#X_train_counts.shape



#applying bag of words algorithm
#segment each text file into words (for English splitting by space), and 
#count # of times each word occurs in each document and finally assign each word an integer id.
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer().fit(temails.body)
X_train_counts = count_vect.transform(temails.body)
#X_train_counts.shape



# TF-IDF i.e Term Frequency times inverse document frequency.
# TF:  Just cWindesheim Honours College "The Holland Scholarship" Challenge
#      it will give more weightage to longer documents than shorter documents.
# IDF: Finally, we can even reduce the weightage of more common words like (the, is, an etc.) 
#      which occurs in all document.
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_tfidf = tfidf_transformer.transform(X_train_counts)
#X_train_tfidf.shape


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, temails.category )
clf.score(X_train_tfidf, temails.category)

while True:
	text = input("Enter Email body").lower()
	X_predict = count_vect.transform(np.array([text]))
	X_predict_tfidf = tfidf_transformer.transform(X_predict)
	#X_predict_tfidf.shape
	print("This Looks like an {}".format(clf.predict(X_predict_tfidf)))
