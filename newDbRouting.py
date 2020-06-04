#
# Authors: Stella, Parthiv, Amir
#


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

  cursor = coll.find({},tailable=True,timeout=False)

  # iterate through db and send to front end

  # example of data to pass to js file 
  data = {'username': 'tweet', 'site': 'stackoverflow.com'}
  return render_template('newHeatmap.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
