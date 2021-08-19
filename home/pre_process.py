import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import re
import string
from nltk.tokenize import word_tokenize

STOPWORDS = set(stopwords.words('english') + list(string.punctuation) + ['covid','pandemic', 'virus', 'corona'])


df = pd.read_csv('finalSentimentData2.csv')

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(num_words = 5000, split = " ")

tokenizer.fit_on_texts(df['text'].values)
X = tokenizer.texts_to_sequences(df['text'].values)
X1 = tokenizer.sequences_to_matrix(X, mode = 'binary')
#X =  pad_sequences(X)
print(len(X1), X1.shape)


#