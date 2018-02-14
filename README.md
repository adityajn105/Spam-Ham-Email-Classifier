# Spam-Ham-Email-Classifier
This is a Flask Web App for classifying emails into Spam and Ham. The algorithm uses various libraries.

The Pipeline is given Below

1. Remove Stop words via Sckit CountVectorizer
2. Stemming Words via NLTK
3. Bag Of Words Algorithm  via Sckit CountVectorizer
4. TF-IDF
5. Multinomial Naive Bayes Classifier

More things to be implemented
1. GridCV


## Installing NLTK via Command Line
Run the command 
```
python -m nltk.downloader all
``` 
To ensure central installation, run the command 
```
sudo python -m nltk.downloader -d /usr/local/share/nltk_data all
```

 You can use the -d flag to specify a different location (but if you do this, be sure to set the NLTK_DATA environment variable accordingly).


## How to run
Required Python Libraries
1. python-flask
2. pandas
3. numpy
3. sckit-learn
4. nltk

To run app:
```
	python app.py
```
Initializing classifier may take about half and hour due to large training dataset So wait patiently .

The app by default will run on port 3000. You can change it from app.py.


## Some ScreenShots

:-------------------------:|:-------------------------:
![Spam 1](https://raw.githubusercontent.com/Adityajn/Spam-Ham-Email-Classifier/master/Screenshots/spam1.png)  		 |  ![Ham 1](https://raw.githubusercontent.com/Adityajn/Spam-Ham-Email-Classifier/master/Screenshots/ham1.png)
:-------------------------:|:-------------------------:
![Spam 2](https://raw.githubusercontent.com/Adityajn/Spam-Ham-Email-Classifier/master/Screenshots/spam2.png)  		 |  ![Ham 2](https://raw.githubusercontent.com/Adityajn/Spam-Ham-Email-Classifier/master/Screenshots/ham2.png)
