from pymongo import MongoClient
import sys
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd 
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
import os
import string




def preprocess(tweet):

    STOPWORDS =  set(stopwords.words('english') + list(string.punctuation) + ['covid','pandemic', 'virus', 'corona'])
    tweet = tweet.lower() # convert to lower case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ' ,tweet) # replace the urls in the link with space
    tweet = re.sub('@[^\s]+',' ',tweet) # replace the @ symabol with space
    tweet = re.sub(r'#([^\s]+)',r'\1',tweet) # replace the # with the word
    tweet = re.sub('[^A-Za-z\s]',' ', tweet) # try and check if its only alphates
    tweet = word_tokenize(tweet) # using NLTK to tokenize
    st = nltk.PorterStemmer() # calling an instance of  porter stemmer
    tweet = [st.stem(word) for word in tweet]
    lm = nltk.WordNetLemmatizer()  #calling an instance of lemmatizer 
    tweet = [lm.lemmatize(word) for word in tweet]
    
    return ' '.join([word for word in tweet if word not in STOPWORDS])

def preprocess_with_len(tweet):
    STOPWORDS =  set(stopwords.words('english') + list(string.punctuation) + ['covid','pandemic', 'virus', 'corona'])
    tweet = tweet.lower() # convert to lower case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ' ,tweet) # replace the urls in the link with space
    tweet = re.sub('@[^\s]+',' ',tweet) # replace the @ symabol with space
    tweet = re.sub(r'#([^\s]+)',r'\1',tweet) # replace the # with the word
    tweet = re.sub('[^A-Za-z\s]',' ', tweet) # try and check if its only alphates
    tweet = word_tokenize(tweet) # using NLTK to tokenize
    tweet = [word for word in tweet if len(word)>2]
    st = nltk.PorterStemmer() # calling an instance of  porter stemmer
    tweet = [st.stem(word) for word in tweet]
    lm = nltk.WordNetLemmatizer()  #calling an instance of lemmatizer 
    tweet = [lm.lemmatize(word) for word in tweet]
    
    return ' '.join([word for word in tweet if word not in STOPWORDS])







df = pd.read_csv('test_data_senti.csv') # read the file

df['preprocessed'] = df['content'].apply(preprocess_with_len) # apply the preprocessing 


vectorizer  = CountVectorizer().fit(df['preprocessed']) # process of assigning vectors for Naive Bayes

train_count = text_vectorizer.transform(X_train) #title_bow.. Needed to repeat this step again

tfidf_transformed = TfidfTransformer().fit(train_count) # to calculate tf-idf

train_tfidf = tfidf_transformed.transform(train_count) # title-tfidf
model_NaiveBayes = MultinomialNB().fit(train_tfidf,df['sentiment'])


model_Random = RandomForestClassifier(n_estimators = 200).fit(text_tfidf,y_train)
predictions_rand = model_Random.predict(test_tfidf)


from sklearn.metrics import accuracy_score
print('Accuracy Score NB: ', accuracy_score(y_test,predictions_naive))
print('Accuracy Score RF: ', accuracy_score(y_test,predictions_rand))



''''
pickl1 ={'vectorizer':text_vectorizer,
         'tfid_transform':tfidf_transformed,
         'model_NaiveBayes': model_NaiveBayes,
         'model_Random': model_Random,
        }

pickle.dump(pickl1, open('models1'+".p", "wb"))
'''