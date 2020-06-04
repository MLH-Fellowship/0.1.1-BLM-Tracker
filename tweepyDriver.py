#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#


from pymongo import MongoClient
from utils import *
from apiKeys import *
from urllib3.exceptions import ProtocolError
importModules = ['sys', 'tweepy', 'json', 'googlemaps', 'time']
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


# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017')
db = client.tweetDatabase
col = db.tweetCollection
#db.command('convertToCapped', 'tweetCollection', size=5242880)
# print("connected to MongoDB:", client["HOST"])

tweetCounter = 0
tweetLimit = 3
trialCounter = 0


class MyStreamListener(tweepy.StreamListener):

    def on_data(self, data):
        global trialCounter, tweetCounter
        trialCounter += 1
        sys.stdout.write("\rTweet #: {}".format(trialCounter))
        sys.stdout.flush()
        obj = json.loads(data)
        location = locationExists(obj)
        if location:
            coordinates = locationIsValid(location)
            if coordinates:
                # if coordinatesInBounds(obj["place"]["bounding_box"]["coordinates"][0][0]):
                store_tweet(obj, col)

                tweetCounter += 1
                print("Location Validated! {}".format(tweetCounter))
                if tweetCounter == tweetLimit:
                    sys.exit(0)

    def on_status(self, status):
        print(status.text)

    def on_error(self, status_code):
        print("Error: {}".format(status_code))
        if status_code == 420:
            rateLimitWait()


def main():
    loadTerms()

    print("\nStarting streaming\n")

    auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret_key)
    auth.set_access_token(twitter_access_token, twitter_access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)

    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
    myStream.disconnect()

    while True:
        try:
            # Temporary filter
            myStream.filter(track=filterTerms, is_async=True)
        except (ProtocolError, AttributeError):
            print("\nIncompleteRead error encountered, continuing\n")
            continue


if __name__ == "__main__":
    main()
