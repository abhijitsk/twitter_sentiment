from django.apps import AppConfig
from django.conf import settings
import os
import pickle
from . import streamer1
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model



class HomeConfig(AppConfig):

    name = 'dbMongo'
    path = os.path.join(settings.MODELS, 'models1.p')
    path1 = os.path.join(settings.MODELS, 'modelsalla.p')
    path2 = os.path.join(settings.MODELS, 'LSTM.h5')

    
    
    with open(path, 'rb') as pickled:
        data = pickle.load(pickled)

    with open(path1, 'rb') as pickled:
        token_LSTM = pickle.load(pickled)

    model_LSTM = load_model(path2)
    




    model_NaiveBayes = data['model_NaiveBayes']
    model_Random = data['model_Random']
    vectorizer = data['vectorizer']
    tfidf_transform = data['tfid_transform']
    tokenize_LSTM = token_LSTM['LSTM_tokenize']


    



