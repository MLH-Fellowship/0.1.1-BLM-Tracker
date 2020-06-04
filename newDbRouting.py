import json
import time
import math
import threading
import signal
import sys
from flask import Flask
from flask import request
from flask import abort
from flask import url_for
from flask import make_response
from flask import Response
from pymongo import MongoClient
from bson import json_util
from threading import Thread

def connect_db():
    client = MongoClient('mongodb://localhost:27017')
    db = client.tweetDatabase
    coll = db.tweetCollection

    cursor = coll.find({},tailable=True,timeout=False)

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








app = Flask(__name__)
@app.route('/tweets')
def tweets():
    url_for('static', filename='map.html')
    url_for('static', filename='jquery-1.7.2.min.js')
    url_for('static', filename='jquery.eventsource.js')
    url_for('static', filename='jquery-1.7.2.js')
    return Response(event_stream(), headers={'Content-Type':'text/event-stream'})

def runThread():
    st = Thread( target = connect_db )
    st.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    app.before_first_request(runThread)
    app.run(debug=True, host='127.0.0.1')
