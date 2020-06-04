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
                if coordinatesInBounds(obj["place"]["bounding_box"]["coordinates"][0][0]):
                    # Replace with database insertion
                    with open("test.json", 'a') as jsonFile:
                        jsonFile.write(json.dumps(obj, indent=4) + "\n")
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
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    myStreamListener = MyStreamListener()
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
    myStream.disconnect()

    while True:
        try:
            myStream.filter(track=filterTerms, is_async=True)  # Temporary filter
        except (ProtocolError, AttributeError):
            print("\nIncompleteRead error encountered, continuing\n")
            continue


if __name__ == "__main__":
    main()
