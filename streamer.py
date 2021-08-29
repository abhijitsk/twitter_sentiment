
import tweepy
from pymongo import MongoClient
import json
import sys


consumer_key = "0KNplHp7ZAPcSP8hJBbFzMUeu"
consumer_secret = "nJFATXys4wL4PcigCfzlc6C5nWTY7IljDFs7ZixKPepBmrJur2"
access_token_key = "99162393-jl6rOgbcTQ1mO3AJCtjafWC87Sr8rQeyhN2z2OpXM"
access_token_secret = "qyxJzAKpEaukzL7xa3Lsnbc1SfhX5lrLMFf06zp9Ctk7T"

MONGO_HOST= 'mongodb+srv://twitter_web:sentiment@cluster0.njeft.mongodb.net/<twitter_test>?retryWrites=true&w=majority'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth)




class MyStreamListener(tweepy.StreamListener):
    def on_status(self,status):
        try:
            client = MongoClient(MONGO_HOST)
            db = client.twitter
            # if status.retweeted:  ### to get the original tweets 
            db.after_.insert_one(status._json)
            print (status.text)


        except :
            print("Error in Status retrieval")
            return False # to end the stream 

    def on_error(self,data):
        print('error 2')
        


if __name__ == "__main__":
    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener, )
    myStream.filter(track=['corona','COVID19','pandemic'],languages = ['en'])

