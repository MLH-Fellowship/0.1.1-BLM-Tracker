#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#


import pandas as pd
from string import digits
import numpy as np

from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence

from tqdm import tqdm
tqdm.pandas()

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.
tokenizer = TweetTokenizer()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale


trainingDataFile = "sentimentAnalysisData/trainingData.csv"
testDataFile = "sentimentAnalysisData/testData.csv"
dimensionCount = 200  # Number of dimensions in which to represent the words
remove_digits = str.maketrans('', '', digits)  # To strip digits
word2Vec = Word2Vec(size=dimensionCount, min_count=10, workers=4)  # Change based on CPU core count
tfidf = dict()


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


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    for word in tokens:
        try:
            vec += word2Vec[word].reshape((1, size)) * tfidf[word]
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


def main():
    # Ingesting training data
    trainingData = ingestData(testDataFile)  # TODO: change to training file
    print("Data loaded with shape: {}\n".format(trainingData.shape))

    # Scrubbing and verifying data
    trainingData = postProcess(trainingData)
    trainingX, trainingY = np.array(trainingData.tokens), np.array(trainingData.Sentiment)
    print("\nLabelling tweets")
    trainingX = labelTokens(trainingX, "TRAIN")

    # Training the word to vector classifier to contextualize relevant words
    global word2Vec
    print("\nConstructing Word to Vector vocabulary:")
    word2Vec.build_vocab([x.words for x in tqdm(trainingX)])
    print("\nTraining Word to Vector classifier:")
    word2Vec.train(sentences=[x.words for x in tqdm(trainingX)], total_examples=len(trainingX), epochs=10)

    # Determining the importance of each word by constructing an tfidf model
    global tfidf
    print("\nConstructing TF-IDF Similarity Matrix:")
    vectorizedTfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
    tfdifMatrix = vectorizedTfidf.fit_transform([x.words for x in tqdm(trainingX)])
    tfidf = dict(zip(vectorizedTfidf.get_feature_names(), vectorizedTfidf.idf_))

    # Vectorize and scale training data
    print("\n Vectorizing and scaling training data:")
    trainingVectors = np.concatenate([buildWordVector(z, dimensionCount) for z in tqdm(map(lambda x: x.words, trainingX))])
    trainingVectors = scale(trainingVectors)

    # Construct sentiment analysis model
    sentimentLSTM = keras.Sequential()
    kerasTokenizer = Tokenizer()
    kerasTokenizer.fit_on_texts(trainingX)
    inputDim = len(kerasTokenizer.word_index) + 1
    # TODO: get length of max tweet
    # TODO: pad tweets
    sentimentLSTM.add(layers.Embedding(input_dim=inputDim, output_dim=256))
    sentimentLSTM.add(layers.GRU(256, dropout=0.2, recurrent_dropout=0.5, return_sequences=True))
    sentimentLSTM.add(layers.GRU(128, dropout=0.2, recurrent_dropout=0.5))
    sentimentLSTM.add(layers.Dense(3, activation='softmax'))
    sentimentLSTM.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    # sentimentLSTMHistory = sentimentLSTM.fit(trainingX, trainingY, validation_split=0.2, epochs=10, batch_size=256)  # 100 epochs
    # print(sentimentLSTMHistory)
    # loss, accuracy = sentimentLSTM.evaluate(trainingX, trainingY, verbose=False)
    # print("Training Accuracy: {:.4f}".format(accuracy))

    print("\nFinished up to this point")  # TODO: Remove

    # print(word2Vec.wv.most_similar('good'))
    # print(trainingData['Sentiment'].value_counts())


if __name__ == "__main__":
    main()
