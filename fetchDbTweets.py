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
  coll = db.tweetCol

  docs = coll.find()
  data = [] # empty array to store lat and long values as for each tweet in dictioary format. Will be an array of dictionaries

  # iterate through db collection and add each document's latitude and longitude
  for item in docs:
    # print(item['place']['bounding_box']['coordinates'][0][0][0])
    s = '{\"lat\": ' + str(item['place']['bounding_box']['coordinates'][0][0][0]) + ', \"long\": ' + str(item['place']['bounding_box']['coordinates'][0][0][1]) + ', \"Sentiment\": ' + str(item['Sentiment'])  + '}'
    data.append(s)

  separator = ', '
  arrayStr = separator.join(data)

  dataStr = '{\"array\":[' + arrayStr + ']}'

  return render_template('heatmap.html', data=dataStr)    # TODO: replace "test.html" with correct html file name


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
