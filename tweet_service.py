#!/bin/env python
# encoding: utf-8

"""
Webservice prototype for exposing cell data via
HTTP and JSON/JSONP
"""


DB = 'tweetDatabase'  # (done) this is just the name of the file that creates a stream, change db name to reflect our db name
CELLS_COLLECTION = 'tweetCollection'


import json
import time
import math
import redis
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


red = redis.StrictRedis()

def signal_handler(signal, frame):
    print ('You pressed Ctrl+C!')
    sys.exit(0)

def tail_mongo_thread():
    print ("beginning to tail...")

    # maybe reestablish db connection here like we did in tstream.py file instead of having tweets stream straight from tweepy
    # new db connection 
    client = MongoClient('mongodb://localhost:27017')
    db = client.tweetDatabase
    coll = db.tweetCollection

    # old db connection
    #db = Connection().tstream   # change db name to reflect our db name
    #coll = db.tweets_tail

    # look into if this coordinate system still stands for our implementation
    cursor = coll.find({},tailable=True,timeout=False)

    ci=0
    while cursor.alive:
        try:
            doc = cursor.next()
            ci += 1
            red.publish('chat', u'%s' % json.dumps(doc,default=json_util.default))
        except StopIteration:
            pass

def event_stream():
    pubsub = red.pubsub()
    pubsub.subscribe('chat')
    i=0
    for message in pubsub.listen():
        i+=1
        print (i)
        yield 'data: %s\n\n' % message['data']

app = Flask(__name__)
@app.route('/tweets')
def tweets():

    url_for('static', filename='map.html')
    url_for('static', filename='jquery-1.7.2.min.js')
    url_for('static', filename='jquery.eventsource.js')
    url_for('static', filename='jquery-1.7.2.js')
    return Response(event_stream(), headers={'Content-Type':'text/event-stream'})

def runThread():
    st = Thread( target = tail_mongo_thread )
    st.start()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    app.before_first_request(runThread)
    app.run(debug=True, host='127.0.0.1')
