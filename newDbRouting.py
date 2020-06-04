import json
import time
import math
import threading
import signal
import sys
from flask import Flask
from flask import render_template
from pymongo import MongoClient
from bson import json_util

app = Flask(__name__)

@app.route("/")
def fetch_tweets():
  client = MongoClient('mongodb://localhost:27017')
  db = client.tweetDatabase
  coll = db.tweetCollection

  # goes through tweets and adds lat and long of tweets into 'data' so 'data' can later be passed into the html file
  docs = coll.find()
  data = [] # empty array to store lat and long values as for each tweet in dictioary format. Will be an array of dictionaries

  # next() is supposed to return something like {'_id': ObjectId('5cda8b3b665444800ad30129'), 'field': 'value'}, which should be the document. As such, feel free to change the variable names below to reflect how our documents look

  # iterate through db collection and add each document's latitude and longitude
  for item in docs:
    # print(item['place']['bounding_box']['coordinates'][0][0][0])
    i = {'lat': item['place']['bounding_box']['coordinates'][0][0][0], 'long': item['place']['bounding_box']['coordinates'][0][0][1]}
    data.append(i)
  
  print(data)
  
  return render_template('test.html', data=data)

# @app.before_first_request(fetch_tweets)
# @app.route("/")
# def main():
#   return render_template('/static/test.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
