#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#


from apiKeys import *
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

gMaps = googlemaps.Client(key=googlemaps_api_key)

rateLimitWaitTime = 1
filterTerms = list()
termFileName = "BLMTerms.txt"

topRight = [49.123606, -65.688553]
bottomLeft = [24.349138, -124.923975]


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


def coordinatesInBounds(coordinates):
    return topRight[0] >= coordinates[1] >= bottomLeft[0] and bottomLeft[1] >= coordinates[0] >= topRight[1]


def locationIsValid(location):
    geocode_result = gMaps.geocode(location)
    print("\nValidating location")
    if geocode_result:
        return True
    print("Validation failed")
    return False


def store_tweet(tw, col):
    col.insert_one(tw)
