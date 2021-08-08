import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import re
import string
from nltk.tokenize import word_tokenize


# load the data
df = pd.read_csv('finalSentimentData2.csv')


class SentimentAnalysis:
    
    
    
    def __init__(self):
        self._stopwords = set(stopwords.words('english') + list(string.punctuation) + ['covid','pandemic', 'virus', 'corona']) # needed to change the hardcoded method
        
        
    def preprocess(self, tweet):
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
        
        
        return  ' '.join([word for word in tweet if word not in self._stopwords])
    
    
    def without_stem(self, tweet):
        tweet = tweet.lower() # convert to lower case
        tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ' ,tweet) # replace the urls in the link with space
        tweet = re.sub('@[^\s]+',' ',tweet) # replace the @ symabol with space
        tweet = re.sub(r'#([^\s]+)',r'\1',tweet) # replace the # with the word
        tweet = re.sub('[^A-Za-z\s]',' ', tweet) # try and check if its only alphates
        tweet = word_tokenize(tweet)
        
        return [word for word in tweet if word not in self._stopwords]
    
    
    def generate_allwords(self,cleaned_data):
        return None
        
    

senti = SentimentAnalysis()


from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df['text'] = df['text'].apply(senti.preprocess)

tokenizer = Tokenizer(num_words = 5000, split = " ")

tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X =  pad_sequences(X)

model = Sequential() # implemending the model
model.add(Embedding(5000,256, input_length = X.shape[1]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, dropout = 0.3, recurrent_dropout = 0.2))
model.add(LSTM(256,dropout = 0.3,recurrent_dropout = 0.2))
model.add(Dense(4, activation ='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

y = pd.get_dummies(df['sentiment']).values


check_label = ['anger','fear','joy','sad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # train test split

batch_size = 32
epochs = 8

model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 2)

predictions = model.predict(X_test)


predictions_label = []
test_labels = []
import numpy as np

for i in range(0,len(predictions)-1):
    predictions_label.append(check_label[np.argmax(predictions[i].round())])
    test_labels.append(check_label[np.argmax(y_test[i].round())])


from sklearn.metrics import accuracy_score
print('Accuracy Score: ', accuracy_score(predictions_label,test_labels))  