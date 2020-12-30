from twitter_credentials import *
import tweepy
from pymongo import MongoClient
import json
import sys


auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth)

Todays_data = []

class MyStreamListener(tweepy.StreamListener):
    def on_status(self,data):
        try:
            client = MongoClient(MONGO_HOST)
            db = client.twitter
            db.twitter_data1.insert_one(json.loads(data))

        except:
            pass

    def on_error(self,data):
        print('error 2')


if __name__ == "__main__":
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)
    myStream.filter(track=['corona','COVID19'],languages = ['en'])

