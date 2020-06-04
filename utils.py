#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#

import sys
import time
import json
import googlemaps
from apiKeys import *

gMaps = googlemaps.Client(key=googlemaps_api_key)

rateLimitWaitTime = 1


def rateLimitWait():
    print("\nRate limit exceeded, waiting for {} seconds\n".format(rateLimitWaitTime))
    for seconds in range(rateLimitWaitTime):
        sys.stdout.write("\rSeconds Remaining: {}".format(rateLimitWaitTime - seconds))
        sys.stdout.flush()
        time.sleep(1)
    print("\n" * 3)


def locationExists(jsonOBJ):
    # print(json.dumps(jsonOBJ, indent=4))
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


def locationIsValid(location):
    geocode_result = gMaps.geocode(location)
    print("Validating location")
    if geocode_result:
        print("Location Validated!: {}".format(geocode_result))
        return geocode_result[0]['geometry']
    print("Validation failed")
    return False
