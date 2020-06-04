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

  #cursor = coll.find({},tailable=True,timeout=False)

  # iterate through db and send to front end

  # ------------------------------ this section is just for testing if everything is running, feel free to comment all this out ------------------------------

  # test to see if connection properly made
  print ("\nReturn every document:")

  #  ---------- iterate and store all IDs in collection
  ids = [] # create empty list for IDs
  for doc in coll.find():
    ids += [doc["_id"]]

  # print out all IDs
  print ("IDs: ", ids)
  print ("total docs: ", len(ids))

  # ---------- get all documents in a collection and put then in a list
  # returns a list of mondodb documents in the form of Python dictionaries
  documents = list(coll.find())

  # test if documents have been retrieved properly
  for doc in documents:
      # access each document's "_id" key
      print ("\ndoc _id:", doc["_id"])

  # ------------------------------ end of testing section ------------------------------


  # ******************* beginning of iterator section *******************

  # section of code that goes through tweets and adds lat and long of tweets into 'data' so 'data' can later be passed into the html file
  docs.coll.find()
  data = [] # empty array to store lat and long values as for each tweet in dictioary format. Will be an array of dictionaries

  # next() is supposed to return something like {'_id': ObjectId('5cda8b3b665444800ad30129'), 'field': 'value'}, which should be the document. As such, feel free to change the variable names below to reflect how our documents look

  # iterate through db collection and add each document's latitude and longitude
  for lat, long in docs.next().items():
      data += {'lat': lat, 'long': long}

  # ******************* end of iterator section *******************


  # example of data to pass to js file
  #data = {'username': 'tweet', 'site': 'stackoverflow.com'}  # I commented this out since the data from a few lines above should be able to be passed in as a parameter. If not then feel free to make adjustments and let me know
  return render_template('newHeatmap.html', data=data)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
