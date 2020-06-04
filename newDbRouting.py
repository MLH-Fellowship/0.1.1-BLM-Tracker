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
from flask import request
from flask import abort
from flask import url_for
from flask import make_response
from flask import Response
from pymongo import MongoClient
from bson import json_util
from threading import Thread

def tail_mongo_thread():
    client = MongoClient('mongodb://localhost:27017')
    db = client.tweetDatabase
    coll = db.tweetCollection

    cursor = coll.find({},tailable=True,timeout=False)

    # iterate through db and send to front end


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

