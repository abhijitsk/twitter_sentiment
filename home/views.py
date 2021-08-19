from django.shortcuts import render
#from . import dbMongo
#from pymongo import MongoClient
from . import streamer_website
import json
import sys
import pandas as pd
#from .apps import HomeConfig
from rest_framework.views import APIView
from django.http import JsonResponse, HttpResponse
from keras.preprocessing.sequence import pad_sequences


#auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
#auth.set_access_token(access_token_key, access_token_secret)

#api = tweepy.API(auth)

#MONGO_HOST = 'mongodb+srv://twitter_web:sentiment@cluster0.njeft.mongodb.net/<twitter_test>?retryWrites=true&w=majority'
#asd = []


#data = pd.read_csv('finalSentimentdata2.csv')
#for i in data['text']:
#    asd.append(i)


# Create your views here.

def home(request):
    
    return render(request,'home.html',{'tweet':streamer_website.tweets_for_website})


def second(request):
    #chart1 = streamer1.plotting()
    wordplot = streamer1.plot_wordcloud()

    return render(request,'second.html',{'tweet':chart1,'wordplot':wordplot})

class call_model(APIView):
    def get(self, request):
        if request.method == 'GET':

            text = asd[2900:]

            
            vector = HomeConfig.vectorizer.transform(text)
            complete_words_Naive = [word for word in HomeConfig.vectorizer.vocabulary_.keys()]
            test_tfidf = HomeConfig.tfidf_transform.transform(vector)
            tokenize_LSTM = HomeConfig.tokenize_LSTM.texts_to_sequences(text)
            complete_words_LSTM = [word for word in HomeConfig.tokenize_LSTM.word_docs.keys()]

            test_lstm = pad_sequences(tokenize_LSTM, maxlen = 62 ) # hard coded the maxlen
            #print(test_lstm)

            prediction = HomeConfig.model_NaiveBayes.predict(test_tfidf)
            prediction_rf = HomeConfig.model_Random.predict(test_tfidf)
            prediction_LSTM = HomeConfig.model_LSTM.predict(test_lstm)
            #print(prediction_LSTM.round())
            predictionLSTM = streamer1.process_forKeras(prediction_LSTM)

            #prediction_LSTM1 = streamer1.process_forKeras(prediction_LSTM.round())
            new_data = pd.DataFrame(text, columns = ['text'])
            new_data['Naive_Bayes'] = prediction.tolist()
            new_data['Random_forrest'] = prediction_rf.tolist()
            new_data['LSTM'] = predictionLSTM

    

            chart1 = streamer1.plotting(prediction.tolist())
            wordplot1 = streamer1.plot_wordcloud(' '.join(complete_words_Naive))

            chart2 = streamer1.plotting(prediction_rf.tolist())
            wordplot2 = streamer1.plot_wordcloud(' '.join(text)) # features names

            chart3 = streamer1.plotting(predictionLSTM)
            wordplot3 = streamer1.plot_wordcloud(' '.join(complete_words_LSTM))

            
            
            response = {'tweet':chart1,
                        'wordplot':wordplot1,
                        'chart2': chart2,
                        'wordplot2':wordplot2,
                        'chart3':chart3,
                        'wordplot3':wordplot3}



            return render(request,'second.html',response)

