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
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument

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
    data = pd.read_csv(filePath, encoding='latin-1')

    # Appending top row to the bottom to add column labels to data
    topRow = pd.DataFrame(data.iloc[0])
    data.append(topRow, ignore_index=True)
    data.columns = ['Sentiment', 'id', 'time', 'client', 'user', 'Tweet']

    # Drop irrelevant columns
    data.drop(['id', 'time', 'client', 'user'], axis=1, inplace=True)

    # Scale down sentiment values and convert them to ints from strings
    data.replace({'Sentiment': {4: 2, 2: 1, 0: 0}}, inplace=True)
    return data


def tokenFilter(token):
    return not (token.startswith('@') or token.startswith('http') or not token.isalpha())


def tweetTokenizer(tweet):
    try:
        tokens = [token.translate(remove_digits).replace('#', '') for token in tokenizer.tokenize(tweet)]
        return list(filter(tokenFilter, tokens))
    except:
        return 'Invalid tweet'


def postProcess(data):
    print("Tokenizing and scrubbing tweets:")
    data['tokens'] = data['Tweet'].progress_map(tweetTokenizer)
    data = data[data.tokens != 'Invalid tweet']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


def labelTokens(tweets, label):
    labeledTokens = list()
    for index, tokens in enumerate(tweets):
        label = "{}_{}".format(label, index)
        labeledTokens.append(TaggedDocument(words=tokens, tags=[label]))
    return labeledTokens


def main():
    # Ingesting training data
    trainingData = ingestData(testDataFile)  # TODO: change to training file
    print("Data loaded with shape: {}\n".format(trainingData.shape))

    # Scrubbing and verifying data
    trainingData = postProcess(trainingData)
    trainingX, trainingY = np.array(trainingData.tokens), np.array(trainingData.Sentiment)
    print("\nLabelling tweets")
    trainingX, trainingY = labelTokens(trainingX, "TRAIN"), labelTokens(trainingY, "TRAIN")

    # Training the word to vector classifier to contextualize relevant words
    word2Vec = Word2Vec(size=200, min_count=10, workers=4)  # Change based on CPU core count
    print("\nConstructing Word to Vector vocabulary:")
    word2Vec.build_vocab([x.words for x in tqdm(trainingX)])
    print("\nTraining Word to Vector classifier:")
    word2Vec.train(sentences=[x.words for x in tqdm(trainingX)], total_examples=len(trainingX), epochs=10)

    print(word2Vec.wv.most_similar('good'))
    # print(trainingData['Sentiment'].value_counts())


if __name__ == "__main__":
    main()
