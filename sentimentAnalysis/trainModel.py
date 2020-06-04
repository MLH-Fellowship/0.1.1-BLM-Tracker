#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#


import pandas as pd
import time
from string import digits
import numpy as np
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class

LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

trainingDataFile = "sentimentAnalysisData/trainingData.csv"
testDataFile = "sentimentAnalysisData/testData.csv"
remove_digits = str.maketrans('', '', digits)  # To strip digits


def ingestData(filePath):
    # Reading data
    print('\nReading data')
    startTime = time.time()
    data = pd.read_csv(filePath, encoding='latin-1')
    print('Finished reading data: {} seconds'.format(time.time() - startTime))

    # Appending top row to the bottom to add column labels to data
    topRow = pd.DataFrame(data.iloc[0])
    data.append(topRow, ignore_index=True)
    data.columns = ['Sentiment', 'id', 'time', 'client', 'user', 'Tweet']

    # Drop irrelevant columns
    data.drop(['id', 'time', 'client', 'user'], axis=1, inplace=True)

    # Scale down sentiment values and convert them to ints from strings
    data['Sentiment'] = data['Sentiment'].replace('4', 2)  # Positive
    data['Sentiment'] = data['Sentiment'].replace('2', 1)  # Neutral
    data['Sentiment'] = data['Sentiment'].replace('0', 0)  # Negative
    print(data['Sentiment'].value_counts())
    print("Data loaded with shape: {}\n".format(data.shape))
    return data


def tokenFilter(token):
    return not (token.startswith('@') or token.startswith('http') or not token.isalpha())


def tweetTokenizer(tweet):
    try:
        tokens = [token.translate(remove_digits).replace('#', '') for token in tokenizer.tokenize(tweet)]
        return list(filter(tokenFilter, tokens))
    except:
        return 'Invalid tweet'


def main():
    trainingData = ingestData(testDataFile)  # TODO: change to training file
    print(tweetTokenizer(trainingData.iloc[1]['Tweet']))


if __name__ == "__main__":
    main()
