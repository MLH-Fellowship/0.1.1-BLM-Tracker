from pymongo import MongoClient
import json
from random import randint
import datetime

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client.tweetDatabase
col = db.tweetCollection
#db.command('convertToCapped', 'tweetCollection', size=5242880)
print("connected to MongoDB:", client["HOST"])


# function that adds one json obj to database collection
def store_tweet(tw):
    col.insert_one(tw)


def iterate_tweetCollection():
    tweets_iterator = col.find()
    return tweets_iterator

    # for tweet in tweets_iterator:
    #    print(tweet['place'])


store_tweet({
    "name": 'stella'
})
