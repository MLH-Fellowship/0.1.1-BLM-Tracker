#
# Created by Parthiv Chigurupati, Stella Wang, and Amir Yalamov
#

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

import pandas as pd
import numpy as np

from tqdm import tqdm

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

tokenizer = TweetTokenizer()
tqdm.pandas()

testDataFile = "sentimentAnalysisData/testData.csv"
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
VALIDATION_SPLIT = 0.2


data = pd.read_csv(testDataFile, encoding='latin-1')

from string import digits

remove_digits = str.maketrans('', '', digits)  # To strip digits

# Appending top row to the bottom to add column labels to data
topRow = pd.DataFrame(data.iloc[0])
data.append(topRow, ignore_index=True)
data.columns = ['Sentiment', 'id', 'time', 'client', 'user', 'Tweet']

# Drop irrelevant columns
data.drop(['id', 'time', 'client', 'user'], axis=1, inplace=True)

# Scale down sentiment values and convert them to ints from strings
data.replace({'Sentiment': {4: 2, 2: 1, 0: 0}}, inplace=True)


def tokenFilter(token):
    return not (token.startswith('@') or token.startswith('http') or not token.isalpha())


def tweetTokenizer(tweet):
    try:
        tokens = [token.translate(remove_digits).replace('#', '') for token in tokenizer.tokenize(tweet)]
        return ' '.join(list(filter(tokenFilter, tokens)))
    except:
        return 'Invalid tweet'


print("\nTokenizing and scrubbing tweets:")
data['tokens'] = data['Tweet'].progress_map(tweetTokenizer)
data = data[data.tokens != 'Invalid tweet']
data.reset_index(inplace=True)
data.drop('index', inplace=True, axis=1)

texts = data['tokens'].to_list()
labels = data['Sentiment'].to_list()
###################################################
print('Found %s texts.' % len(texts))

# finally, vectorize the text samples into a 2D integer tensor
kerasTokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
kerasTokenizer.fit_on_texts(texts)
sequences = kerasTokenizer.texts_to_sequences(texts)

word_index = kerasTokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:]
y_train = labels[:]
# x_val = data[-num_validation_samples:]
# y_val = labels[-num_validation_samples:]

###################################################

# load model
model = load_model('model.h5')
# summarize model.
model.summary()
score = model.evaluate(x_train, y_train, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

for x in range(10):
    print(y_train[x], model.predict(np.array([x_train[x]])))