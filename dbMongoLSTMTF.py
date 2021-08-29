import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from gensim.models import Word2Vec

STOPWORDS =  set(stopwords.words('english') + list(string.punctuation) + ['covid','pandemic', 'virus', 'corona'])
def preprocess(tweet):
    tweet = tweet.lower() # convert to lower case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ' ,tweet) # replace the urls in the link with space
    tweet = re.sub('@[^\s]+',' ',tweet) # replace the @ symabol with space
    tweet = re.sub(r'#([^\s]+)',r'\1',tweet) # replace the # with the word
    tweet = re.sub('[^A-Za-z\s]',' ', tweet) # try and check if its only alphates
    tweet = word_tokenize(tweet) # using NLTK to tokenize


    return [word for word in tweet if word not in STOPWORDS]


# reaing the database file
df_x = pd.read_csv('test_data_senti.csv')
df_x['prepocessed'] = df_x['content'].apply(preprocess)


### loadingWord2vec Model



pre_processed_data = df_x['content'].values

word2vecModel = Word2Vec(pre_processed_data, vector_size = 256, min_count = 1, workers = 3, window =3,sg=1)
weights_embd = word2vecModel.wv.vectors

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


tokenizer = Tokenizer(split = " ")
tokenizer.fit_on_texts(df_x['content'].values)
X = tokenizer.texts_to_sequences(df_x['content'].values)
X =  pad_sequences(X, padding = 'post')

embeddings = np.zeros((len(tokenizer.word_index)+1, 256))

for word , i in tokenizer.word_index.items():
  try:
    embedding_vector = word2vecModel.wv.get_vector(word)
    if embedding_vector is not None:
      embeddings[i] =embedding_vector
  except:
    pass

model = Sequential()
model.add(Embedding(24734,256, trainable = False, weights = [embeddings]))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True, dropout = 0.3, recurrent_dropout = 0.2))
model.add(LSTM(256,dropout = 0.3,recurrent_dropout = 0.2))
model.add(Dense(4, activation ='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])


y = pd.get_dummies(df_x['sentiment']).values
check_label = ['worry','neutral','happiness','sadness']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
batch_size = 32
epochs = 60
model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, verbose = 2)

predictions = model.save('modelLS1.h5')
predictions = model.predict(X_test)

predictions_label = []   # 20,000 features accuracy 42; 15,000 features
results =[]

for i in range(0,len(predictions)-1):
    predictions_label.append(check_label[np.argmax(y_test[i].round())])
    results.append(check_label[np.argmax(predictions[i].round())])

from sklearn.metrics import accuracy_score, confusion_matrix
print('Accuracy Score: ', accuracy_score(predictions_label,results))




