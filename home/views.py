from django.shortcuts import render
#from . import dbMongo
#from pymongo import MongoClient
from . import streamer1
import json
import sys
import pandas as pd
from .apps import HomeConfig
from rest_framework.views import APIView
from django.http import JsonResponse



#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token_key, access_token_secret)

#api = tweepy.API(auth)

#MONGO_HOST = 'mongodb+srv://twitter_web:sentiment@cluster0.njeft.mongodb.net/<twitter_test>?retryWrites=true&w=majority'
asd = []


data = pd.read_csv('finalSentimentdata2.csv')
for i in data['text']:
    asd.append(i)


# Create your views here.

def home(request):
    
    return render(request,'home.html',{'tweet':asd})


def second(request):
    chart1 = streamer1.plotting()
    wordplot = streamer1.plot_wordcloud()

    return render(request,'second.html',{'tweet':chart1,'wordplot':wordplot})

class call_model(APIView):
    def get(self, request):
        if request.method == 'GET':

            text = asd[2900:]


            vector = HomeConfig.vectorizer.transform(text)

            test_tfidf = HomeConfig.tfidf_transform.transform(vector)

            prediction = HomeConfig.model.predict(test_tfidf).tolist()
            print(prediction)



            response = {'text_sentiment': prediction}

            return JsonResponse(response)