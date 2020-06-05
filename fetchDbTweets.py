#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#


import json
import time
import math
import threading
import signal
import sys
from flask import Flask, render_template, Response, request, redirect, url_for
from pymongo import MongoClient
from bson import json_util

app = Flask(__name__)

@app.route("/")
@app.route("/refresh/", methods=['POST'])
def fetch_tweets():
  client = MongoClient('mongodb://localhost:27017')
  db = client.tweetDatabase
  coll = db.sentimentTweets

  docs = coll.find()
  data = [] 

  # iterate through db collection and add each document's latitude and longitude
  for item in docs:
    s = '{\"lat\": ' + str(item['place']['bounding_box']['coordinates'][0][0][0]) + ', \"long\": ' + str(item['place']['bounding_box']['coordinates'][0][0][1]) + ', \"Sentiment\": ' + str(item['Sentiment'])  + '}'
    # s = '{\"lat\": ' + str(item['place']['bounding_box']['coordinates'][0][0][0]) + ', \"long\": ' + str(item['place']['bounding_box']['coordinates'][0][0][1]) + '}'
    data.append(s)

  separator = ', '
  arrayStr = separator.join(data)

  dataStr = '{\"array\":[' + arrayStr + ']}'

  return render_template('heatmap.html', data=dataStr)    # TODO: replace "test.html" with correct html file name


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
