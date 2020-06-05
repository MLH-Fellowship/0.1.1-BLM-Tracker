#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#
from __future__ import print_function

importModules = ['sys', 'tweepy', 'json', 'googlemaps', 'time', 'langdetect']
for module in importModules:
    try:
        globals()[module] = __import__(module)
    except ImportError:
        print("'{}' was not successfully imported, please install '{}' and try again".format(module, module))
        quit()
    except Exception as exception:
        print("{} exception was thrown when trying to import '{}'".format(exception, module))
        quit()

import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from string import digits
remove_digits = str.maketrans('', '', digits)

from apiKeys import *

gMaps = googlemaps.Client(key=googlemaps_api_key)
langdetect.DetectorFactory.seed = 0

rateLimitWaitTime = 1
filterTerms = list()
termFileName = "BLMTerms.txt"

topRight = [49.123606, -65.688553]
bottomLeft = [24.349138, -124.923975]

dataFolder = 'sentimentAnalysis/sentimentAnalysisData'
maxTweetLength = 1000
maxWordCount = 20000
dimensionCount = 100
trainingValidationSplit = 0.2

dataFile = "trainingData.csv"
embeddingsFile = "glove.6B.100d.txt"
modelName = 'model.h5'
invalidTweet = 'Invalid tweet'

model = load_model('sentimentAnalysis/model.h5')


def tokenFilter(token):
    return not (token.startswith('@') or token.startswith('http') or not token.isalpha())


def tweetTokenizer(tweet):
    try:
        tokens = [token.translate(remove_digits).replace('#', '') for token in tokenizer.tokenize(tweet)]
        return ' '.join(list(filter(tokenFilter, tokens)))
    except:
        return invalidTweet


def kerasTweet(tweet):
    kerasTokenizer = Tokenizer(num_words=maxWordCount)
    kerasTokenizer.fit_on_texts([tweet])
    sequences = kerasTokenizer.texts_to_sequences([tweet])
    tweet = pad_sequences(sequences, maxlen=maxTweetLength)
    return tweet


def getSentiment(tweetText):
    tokenizedTweet = tweetTokenizer(tweetText)
    if tokenizedTweet == invalidTweet:
        return -1
    print(tokenizedTweet)
    tweet = kerasTweet(tokenizedTweet)
    sentiment = model.predict(np.array(tweet))
    print(sentiment)
    return sentiment



def loadTerms():
    global filterTerms
    termFile = open(termFileName, "r+")
    tmpTerms = termFile.readlines()
    for term in tmpTerms:
        filterTerms.append(term)
        filterTerms.append("#" + term.strip().lower().replace(" ", ""))
    print("\n" * 2)


def rateLimitWait():
    time.sleep(rateLimitWaitTime)


def locationExists(jsonOBJ):
    if "limit" in jsonOBJ:
        rateLimitWait()
        return False
    if jsonOBJ['place'] != 'null':
        return jsonOBJ['place']
    elif jsonOBJ['user']['location'] != 'null':
        return jsonOBJ['user']['location']
    elif jsonOBJ['retweeted_status'] != 'null' and jsonOBJ['retweeted_status']['place'] != 'null':
        return jsonOBJ['retweeted_status']['place']
    return False


def coordinatesInBounds(coordinates):
    return topRight[0] >= coordinates[0] >= bottomLeft[0] and bottomLeft[1] >= coordinates[1] >= topRight[1]


def isEnglish(jsonOBJ):
    if jsonOBJ['lang'] == 'en':
        return True
    tweetText = jsonOBJ['extended_tweet']['full_text'] if jsonOBJ['truncated'] else jsonOBJ['quoted_status']['text']
    if langdetect.detect(tweetText) == 'en':
        return True
    confidences = langdetect.detect_langs(tweetText)
    for confidence in confidences:
        if confidence.prob > 95:  # 95% is two standard deviations on a normal curve
            if confidence.lang == 'en':
                return True
        else:
            break
    return False


def locationIsValid(location):
    geocode_result = gMaps.geocode(location)
    print("\nValidating location")
    if geocode_result:
        return geocode_result[0]
    print("Validation failed")
    return False
