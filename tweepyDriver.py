#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#


import tweepy
from twitterAPIKeys import *
import json
import sys


tweetCounter = 0
tweetLimit = 5
tweets = list()

class MyStreamListener(tweepy.StreamListener):

    def on_data(self, data):
        obj = json.loads(data)
        tweets.append(obj)
        global tweetCounter
        tweetCounter += 1
        print(tweetCounter)
        if tweetCounter == tweetLimit:
            writeJson()
            sys.exit(0)
        return True

    def on_status(self, status):
        print(status.text)

    def on_error(self, status_code):
        print("Error: {}".format(status_code))
        # Rate limit exceeded, backoff and wait
        if status_code == 420:
            return False


def writeJson():
    with open("test.json", 'w') as jsonFile:
        for elem in tweets:
            jsonFile.write(json.dumps(elem, indent = 4) + "\n")
      

auth = tweepy.OAuthHandler(apiKey, apiSecretKey)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit=True)

myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth = api.auth, listener=myStreamListener)

myStream.filter(track=['black'], is_async=True)
