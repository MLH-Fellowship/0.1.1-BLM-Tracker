from __future__ import print_function

import os
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.initializers import Constant

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

from tqdm import tqdm
tqdm.pandas()

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from string import digits
remove_digits = str.maketrans('', '', digits)

dataFolder = 'sentimentAnalysisData'
maxTweetLength = 1000
maxWordCount = 20000
dimensionCount = 100
trainingValidationSplit = 0.2

dataFile = "trainingData.csv"
embeddingsFile = "glove.6B.100d.txt"
modelName = 'model.h5'

embeddingsIndex = dict()
data = pd.read_csv((os.path.join(dataFolder, dataFile)), encoding='latin-1')


def generateEmbeddingsIndex():
    global embeddingsIndex
    print("\nGenerating embeddings index:")
    wordVectorCount = 0
    with open(os.path.join(dataFolder, embeddingsFile)) as f:
        wordVectorCount += len(f.readlines())
    with open(os.path.join(dataFolder, embeddingsFile)) as f:
        for line in tqdm(f, total=wordVectorCount):
            word, vector = line.split(maxsplit=1)
            vector = np.fromstring(vector, 'f', sep=' ')
            embeddingsIndex[word] = vector
    print('Found {} word vectors'.format(wordVectorCount))


def ingestData():
    global data
    print('\nIngesting tweet dataset')

    # Appending top row to the bottom to add column labels to data
    topRow = pd.DataFrame(data.iloc[0])
    data.append(topRow, ignore_index=True)
    data.columns = ['Sentiment', 'id', 'time', 'client', 'user', 'Tweet']

    # Drop irrelevant columns
    data.drop(['id', 'time', 'client', 'user'], axis=1, inplace=True)

    # Scale down sentiment values and convert them to ints from strings
    data.replace({'Sentiment': {4: 2, 2: 1, 0: 0}}, inplace=True)
    print("Ingested tweet dataset")


def tokenFilter(token):
    return not (token.startswith('@') or token.startswith('http') or not token.isalpha())


def tweetTokenizer(tweet):
    try:
        tokens = [token.translate(remove_digits).replace('#', '') for token in tokenizer.tokenize(tweet)]
        return ' '.join(list(filter(tokenFilter, tokens)))
    except:
        return 'Invalid tweet'


def postProcess():
    global data
    print("\nTokenizing and scrubbing tweets:")
    data['tokens'] = data['Tweet'].progress_map(tweetTokenizer)
    data = data[data.tokens != 'Invalid tweet']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data['tokens'].to_list(), data['Sentiment'].to_list()


def tokenizeSequences(texts, labels):
    global data, maxTweetLength
    print('\nScrubbed {} texts'.format(len(texts)))

    # Tokenizing tweets
    kerasTokenizer = Tokenizer(num_words=maxWordCount)
    kerasTokenizer.fit_on_texts(texts)

    # Converting tweets to sequences
    sequences = kerasTokenizer.texts_to_sequences(texts)

    print((max(list(map(len, sequences)))), maxTweetLength)
    # maxTweetLength = min((max(list(map(len, sequences)))), maxTweetLength)

    # Padding sequences to make them equal length
    data = pad_sequences(sequences, maxlen=maxTweetLength)

    wordCount = kerasTokenizer.word_index
    print('Found {} unique tokens'.format(len(wordCount)))

    # Making labels categorical
    labels = to_categorical(np.asarray(labels))

    return wordCount, labels


def prepareKerasData(labels):
    global data

    # Split data into training and training validation data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    validationSampleCount = int(trainingValidationSplit * data.shape[0])

    # Splitting data into inputs and outputs
    trainingX = data[:-validationSampleCount]
    trainingY = labels[:-validationSampleCount]
    validationX = data[-validationSampleCount:]
    validationY = labels[-validationSampleCount:]
    return trainingX, trainingY, validationX, validationY


def generateEmbeddingsMatrix(wordDict):
    print('\nPreparing embedding matrix.')
    uniqueWordCount = min(maxWordCount, len(wordDict) + 1)
    embeddingsMatrix = np.zeros((uniqueWordCount, dimensionCount))
    for word, i in wordDict.items():
        if i >= maxWordCount:
            continue
        embeddingsVector = embeddingsIndex.get(word)
        if embeddingsVector is not None:
            embeddingsMatrix[i] = embeddingsVector
    return uniqueWordCount, embeddingsMatrix


def getAvailableGPUS():
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


def main():
    tfback._get_available_gpus = getAvailableGPUS
    generateEmbeddingsIndex()
    ingestData()
    texts, labels = postProcess()
    wordDict, labels = tokenizeSequences(texts, labels)
    trainingX, trainingY, validationX, validationY = prepareKerasData(labels)
    uniqueWordCount, embeddingsMatrix = generateEmbeddingsMatrix(wordDict)

    embeddingLayer = Embedding(uniqueWordCount, dimensionCount, embeddings_initializer=Constant(embeddingsMatrix), input_length=maxTweetLength, trainable=False)

    sequenceInput = Input(shape=(maxTweetLength,), dtype='int32')
    embeddedSequences = embeddingLayer(sequenceInput)
    x = Conv1D(128, 5, activation='relu')(embeddedSequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    sentimentClassifier = Dense(3, activation='softmax')(x)

    model = Model(sequenceInput, sentimentClassifier)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    print("\nTraining model: ")
    model.fit(trainingX, trainingY, batch_size=128, epochs=15, validation_data=(validationX, validationY))

    model.save(modelName)
    print("\nModel file saved as {}, training complete\n".format(modelName))


if __name__ == '__main__':
    main()
