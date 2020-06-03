from pymongo import MongoClient
import json
from random import randint

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client.tweetDatabase
test = db.testCollection
col = db.tweetCollection
print("connected to MongoDB:", client["HOST"])


# function that adds one json obj to database collection
def store_tweet(tw):
    test.insert_one(tw)


# sample data for testing connection to db
names = ['Kitchen', 'Animal', 'State', 'Tastey', 'Big', 'City',
         'Fish', 'Pizza', 'Goat', 'Salty', 'Sandwich', 'Lazy', 'Fun']
company_type = ['LLC', 'Inc', 'Company', 'Corporation']
company_cuisine = ['Pizza', 'Bar Food', 'Fast Food',
                   'Italian', 'Mexican', 'American', 'Sushi Bar', 'Vegetarian']
for x in range(1, 51):
    business = {
        'name': names[randint(0, (len(names)-1))] + ' ' + names[randint(0, (len(names)-1))] + ' ' + company_type[randint(0, (len(company_type)-1))],
        'rating': randint(1, 5),
        'cuisine': company_cuisine[randint(0, (len(company_cuisine)-1))]
    }
    # add obj to db here
    store_tweet(business)

print('finished creating 500 business reviews')


# iterate through entire collection
tweets_iterator = test.find()
for tweet in tweets_iterator:
    print(tweet['name'])
