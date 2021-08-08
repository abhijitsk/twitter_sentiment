from django.apps import AppConfig
from django.conf import settings
import os
import pickle
from dbMongo import *

print(2)
#preprocess = SentimentAnalysis().preprocess

class HomeConfig(AppConfig):

    name = 'dbMongo'
    path = os.path.join(settings.MODELS, 'models.p')

    preprocess = SentimentAnalysis().preprocess
    with open(path, 'rb') as pickled:
        data = pickle.load(pickled)
    model = data['model']
    vectorizer = data['vectorizer']
    tfidf_transform = data['tfid_transform']
