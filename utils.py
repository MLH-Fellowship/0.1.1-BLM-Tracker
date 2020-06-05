#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#
from __future__ import print_function

importModules = ['sys', 'tweepy', 'json', 'googlemaps', 'time', 'langdetect']
for module in importModules:
    try:
        globals()[module] = __import__(module)
    except ImportError:
        print("'{}' was not successfully imported, please install '{}' and try again".format(
            module, module))
        quit()
    except Exception as exception:
        print("{} exception was thrown when trying to import '{}'".format(
            exception, module))
        quit()

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from string import digits
remove_digits = str.maketrans('', '', digits)

from userAPIKeys import *


gMaps = googlemaps.Client(key=googlemaps_api_key)
langdetect.DetectorFactory.seed = 0

rateLimitWaitTime = 1
filterTerms = list()
termFileName = "BLMTerms.txt"

topRight = [49.123606, -65.688553]
bottomLeft = [24.349138, -124.923975]
maxSentiment = 9

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


def calculateSentiment(tweetText):
    tokenizedTweet = tweetTokenizer(tweetText)
    if tokenizedTweet == invalidTweet:
        return 0
    tweet = kerasTweet(tokenizedTweet)
    sentiment = model.predict(np.array(tweet))
    return sentiment[0]


def getSentiment(tweet):
    tweetText = tweet['extended_tweet']['full_text'] if tweet['truncated'] else tweet['text']
    sentiment = calculateSentiment(tweetText).tolist()

    # Use sentiment analysis to get a base sentiment
    baseSentiment = 4 - sentiment.index(max(sentiment))

    # Use interactivity to calculate sentiment scalar
    quoteCount = tweet['quote_count']
    replyCount = tweet['reply_count']
    retweetCount = tweet['retweet_count']
    favoriteCount = tweet['favorite_count']
    sentimentScalar = (favoriteCount + retweetCount * 3 + replyCount * 1.5 + quoteCount * 4.5) / 100000

    finalSentiment = min(maxSentiment, max(baseSentiment * sentimentScalar, baseSentiment))
    return finalSentiment


#########################################################################


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
    return topRight[0] >= coordinates[1] >= bottomLeft[0] and bottomLeft[1] >= coordinates[0] >= topRight[1]


def isEnglish(jsonOBJ):
    if jsonOBJ['lang'] == 'en':
        return True
    tweetText = jsonOBJ['extended_tweet']['full_text'] if jsonOBJ['truncated'] else jsonOBJ['text']
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
        return True
    print("Validation failed")
    return False


def store_tweet(tw, col):
    col.insert_one(tw)
