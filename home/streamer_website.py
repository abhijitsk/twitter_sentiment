
import tweepy
from pymongo import MongoClient
import json
import sys
from .twitter_credentials import *


tweets_for_website = []

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)

api = tweepy.API(auth)
for tweet in tweepy.Cursor(api.search, q = ("#corona"),count = 100, lang= 'en').items(100):
        tweets_for_website.append(tweet.text)
