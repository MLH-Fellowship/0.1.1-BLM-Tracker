#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#


import pandas as pd
from string import digits
import numpy as np

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, GRU
from keras.models import Model
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
import keras.backend.tensorflow_backend as tfback


from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()


trainingDataFile = "sentimentAnalysisData/trainingData.csv"
testDataFile = "sentimentAnalysisData/testData.csv"
embeddingsFile = "sentimentAnalysisData/glove.6B.100d.txt"
dimensionCount = 100  # Number of dimensions in which to represent the words
remove_digits = str.maketrans('', '', digits)  # To strip digits
maxWordCount = 3000
embeddingsIndex, embeddingsMatrix = dict(), dict()
tweets, labels = list(), list()


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
        return ' '.join(list(filter(tokenFilter, tokens)))
    except:
        return 'Invalid tweet'


def postProcess(data):
    print("\nTokenizing and scrubbing tweets:")
    data['tokens'] = data['Tweet'].progress_map(tweetTokenizer)
    data = data[data.tokens != 'Invalid tweet']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


def labelTokens(data, label):
    labeledTokens = list()
    for index, tokens in enumerate(data):
        label = "{}_{}".format(label, index)
        labeledTokens.append(TaggedDocument(words=tokens, tags=[label]))
    return labeledTokens


def generateEmbeddingsIndex():
    global embeddingsIndex
    with open(embeddingsFile) as f:
        for line in tqdm(f, total=400000):
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddingsIndex[word] = coefs


def generateEmbeddingsMatrix(trainingData):
    global tweets
    kerasTokenizer = Tokenizer(num_words=maxWordCount)
    kerasTokenizer.fit_on_texts(tweets)
    uniqueWordCount = len(kerasTokenizer.word_index) + 1
    tweetArray = kerasTokenizer.texts_to_sequences(tweets)
    maxTweetLength = max(list(map(len, tweetArray)))
    tweets = pad_sequences(tweetArray, maxlen=maxTweetLength)

    global embeddingsMatrix
    num_words = min(maxWordCount, uniqueWordCount)
    embeddingsMatrix = np.zeros((num_words, dimensionCount))
    word_index = kerasTokenizer.word_index
    for word, i in word_index.items():
        if i >= maxWordCount:
            continue
        embedding_vector = embeddingsIndex.get(word)
        if embedding_vector is not None:
            embeddingsMatrix[i] = embedding_vector

    return uniqueWordCount, maxTweetLength


def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


def main():
    tfback._get_available_gpus = _get_available_gpus

    # Ingesting training data
    trainingData = ingestData(testDataFile)  # TODO: change to training file
    trainingData = postProcess(trainingData)
    global tweets, labels
    tweets = trainingData['tokens'].to_list()
    labels = trainingData['Sentiment'].to_list()
    labels = to_categorical(np.asarray(labels))
    testData = ingestData(testDataFile)
    testData = postProcess(testData)
    testX, testY = np.array(testData.tokens), to_categorical(np.asarray(testData.Sentiment))

    # Generating embeddings matrix
    print("\nGenerating embeddings index:")
    generateEmbeddingsIndex()

    # Scrubbing and verifying data
    uniqueWordCount, maxLength = generateEmbeddingsMatrix(trainingData.Sentiment)

    # Construct sentiment analysis model
    sequence_input = Input(shape=(maxLength,), dtype='int32')
    embedding_layer = Embedding(uniqueWordCount, dimensionCount, embeddings_initializer=Constant(embeddingsMatrix), input_length=maxLength, trainable=False)
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu', data_format='channels_first')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu', data_format='channels_first')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu', data_format='channels_first')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    model.fit(tweets, labels, batch_size=128, epochs=10, validation_data=(testX, testY))

    model.save('model.h5')


if __name__ == "__main__":
    main()
