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

from tqdm import tqdm
tqdm.pandas()

from nltk.tokenize import TweetTokenizer
tokenizer = TweetTokenizer()

from string import digits
remove_digits = str.maketrans('', '', digits)

BASE_DIR = 'sentimentAnalysisData'
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

dataFile = "testData.csv"  # TODO: Change
embeddingsFile = "glove.6B.100d.txt"
modelName = 'model.h5'

embeddings_index = dict()
data = pd.read_csv((os.path.join(BASE_DIR, dataFile)), encoding='latin-1')


def generateEmbeddingsIndex():
    global embeddings_index
    print("\nGenerating embeddings index:")
    wordVectorCount = 0
    with open(os.path.join(BASE_DIR, embeddingsFile)) as f:
        wordVectorCount += len(f.readlines())
    with open(os.path.join(BASE_DIR, embeddingsFile)) as f:
        for line in tqdm(f, total=wordVectorCount):
            word, vector = line.split(maxsplit=1)
            vector = np.fromstring(vector, 'f', sep=' ')
            embeddings_index[word] = vector
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
    global data
    print('\nScrubbed {} texts'.format(len(texts)))

    # Tokenizing tweets
    kerasTokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    kerasTokenizer.fit_on_texts(texts)

    # Converting tweets to sequences
    sequences = kerasTokenizer.texts_to_sequences(texts)

    # Padding sequences to make them equal length
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

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
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    # Splitting data into inputs and outputs
    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    return x_train, y_train, x_val, y_val


def generateEmbeddingsMatrix(word_index):
    print('\nPreparing embedding matrix.')
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return num_words, embedding_matrix


def main():
    generateEmbeddingsIndex()
    ingestData()
    texts, labels = postProcess()
    word_index, labels = tokenizeSequences(texts, labels)
    x_train, y_train, x_val, y_val = prepareKerasData(labels)
    num_words, embedding_matrix = generateEmbeddingsMatrix(word_index)

    embedding_layer = Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), input_length=MAX_SEQUENCE_LENGTH, trainable=False)

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(3, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

    print("\nTraining model: ")
    model.fit(x_train, y_train, batch_size=128, epochs=15, validation_data=(x_val, y_val))

    model.save(modelName)
    print("\nModel file saved as {}, training complete\n".format(modelName))


if __name__ == '__main__':
    main()
