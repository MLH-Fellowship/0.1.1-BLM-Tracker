#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#

from utils import *
import json

filePath = "sentimentAnalysis/sentimentAnalysisData/tweetCollection.json"


def main():
    f = open(filePath)
    tweets = json.load(f)
    for index in range(len(tweets)):
        tweets[index]['Sentiment'] = getSentiment(tweets[index])
        if index % 100 == 0:
            print(index, tweets[index]['Sentiment'])
    with open('sentimentTweets.json', 'w', encoding='utf-8') as f:
        json.dump(tweets, f, ensure_ascii=False, indent=4)
    print("Finished!")


if __name__ == '__main__':
    main()
