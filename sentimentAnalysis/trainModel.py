#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#


import pandas as pd
import time
import numpy as np
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the word2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence # we'll talk about this down below

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

trainingDataFile = "sentimentAnalysisData/trainingData.csv"
testDataFile = "sentimentAnalysisData/testData.csv"


def ingestData(filePath):
    print('\nReading data')
    data = pd.read_csv(filePath, encoding='latin-1')
    print('Finished reading data')
    print(data.isnull().sum())
    data.columns = ['Sentiment', 'id', 'time', 'client', 'user', 'Tweet']
    data.drop(['id', 'time', 'client', 'user'], axis=1, inplace=True)
    data['Sentiment'] = data['Sentiment'].map(int)
    print("Data loaded with shape: {}\n".format(data.shape))
    return data


def main():
    trainingData = ingestData(trainingDataFile)  # TODO: change to training file



if __name__ == "__main__":
    main()
