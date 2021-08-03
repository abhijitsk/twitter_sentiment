from django.shortcuts import render
from . import dbMongo, streamer1
import tweepy
from pymongo import MongoClient
import json
import sys

consumer_key = "0KNplHp7ZAPcSP8hJBbFzMUeu"
consumer_secret = "nJFATXys4wL4PcigCfzlc6C5nWTY7IljDFs7ZixKPepBmrJur2"
access_token_key = "99162393-jl6rOgbcTQ1mO3AJCtjafWC87Sr8rQeyhN2z2OpXM"
access_token_secret = "qyxJzAKpEaukzL7xa3Lsnbc1SfhX5lrLMFf06zp9Ctk7T"


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth)

MONGO_HOST = 'mongodb+srv://twitter_web:sentiment@cluster0.njeft.mongodb.net/<twitter_test>?retryWrites=true&w=majority'


class MyStreamListener(tweepy.StreamListener):
    def on_status(self,status):
        try:
            client = MongoClient(MONGO_HOST)
            db = client.twitter
            # if status.retweeted:  ### to get the original tweets 
            db.Newtocollate1.insert_one(status._json)
            print (type(status._json))


        except :
            print("Error in Status retrieval")
            return False # to end the stream 

    def on_error(self,data):
        print('error 2')

# Create your views here.

def home(request):
    tweetext = dbMongo.listForWeb
    return render(request,'home.html',{'tweet':tweetext})


def second(request):
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    chart1 = plotting(x,y)

    return render(request,'second.html',{'tweet':chart1})