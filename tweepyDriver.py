#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#


importModules = ['sys', 'tweepy', 'json', 'googlemaps', 'time']
for module in importModules:
    try:
        globals()[module] = __import__(module)
    except ImportError:
        print("'{}' was not successfully imported, please install '{}' and try again".format(module, module))
        quit()
    except Exception as exception:
        print("{} exception was thrown when trying to import '{}'".format(exception, module))
        quit()

from urllib3.exceptions import ProtocolError
from apiKeys import *
from utils import *

tweetCounter = 0
tweetLimit = 30
tweets = list()
trialCounter = 0


class MyStreamListener(tweepy.StreamListener):

    def on_data(self, data):
        global trialCounter
        trialCounter += 1
        print(trialCounter)
        obj = json.loads(data)
        location = locationExists(obj)
        if location:
            coordinates = locationIsValid(location)
            if coordinates:
                # replace with database insertions
                tweets.append(obj)
                # for testing
                global tweetCounter
                tweetCounter += 1
                print(tweetCounter)
                if tweetCounter == tweetLimit:
                    writeJson()
                    sys.exit(0)

    def on_status(self, status):
        print(status.text)

    def on_error(self, status_code):
        print("Error: {}".format(status_code))
        if status_code == 420:
            rateLimitWait()


def writeJson():
    with open("test.json", 'w') as jsonFile:
        for elem in tweets:
            jsonFile.write(json.dumps(elem, indent=4) + "\n")


def main():
    auth = tweepy.OAuthHandler(twitter_api_key, twitter_api_secret_key)
    auth.set_access_token(twitter_access_token, twitter_access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)

    while True:
        try:
            myStream.filter(track=['black'], is_async=True)  # Temporary filter
        except (ProtocolError, AttributeError):
            print("\nIncompleteRead error encountered, continuing\n")
            continue


if __name__ == "__main__":
    main()
