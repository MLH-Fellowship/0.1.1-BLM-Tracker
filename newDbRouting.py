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

<<<<<<< HEAD
def connect_db():
    client = MongoClient('mongodb://localhost:27017')
    db = client.tweetDatabase
    coll = db.tweetCollection
=======
>>>>>>> 4bb08794c2fc62fe015a67d19e55b4861867eb6e

app = Flask(__name__)

<<<<<<< HEAD
    # ---------- iterate through db and send to front end

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






=======
@app.route("/")
def fetch_tweets():
  client = MongoClient('mongodb://localhost:27017')
  db = client.tweetDatabase
  coll = db.tweetCollection
>>>>>>> 4bb08794c2fc62fe015a67d19e55b4861867eb6e

  cursor = coll.find({},tailable=True,timeout=False)

  # iterate through db and send to front end

<<<<<<< HEAD
def runThread():
    st = Thread( target = connect_db )
    st.start()
=======
  # example of data to pass to js file 
  data = {'username': 'tweet', 'site': 'stackoverflow.com'}
  return render_template('newHeatmap.html', data=data)
>>>>>>> 4bb08794c2fc62fe015a67d19e55b4861867eb6e


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1')
