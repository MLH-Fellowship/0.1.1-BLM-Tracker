# BLM Tracker
Stream twitter data through a classifier to determine the amount of social activity surrounding a social movement and visualizing it.

## Technologies Used

### Open Source 

* [Blackbox](https://github.com/StackExchange/blackbox)
    * An open-source tool used for file encryption (specifically the API keys)
* [Flask](https://github.com/pallets/flask)
    * A Python microservice used for building and deploying web applications
* [Keras](https://github.com/keras-team/keras)
    * A neural network API running on top of other neural network frameworks (in this case TensorFlow)
* [langdetect](https://github.com/Mimino666/langdetect)
    * A port to Python of Google's language-detection library
* [MongoDB](https://github.com/mongodb/mongo)
    * Database used to store tweets
* [nltk](https://github.com/nltk/nltk)
    * Natural Language Toolkit used to tokenize tweets for word analysis
* [NumPy](https://github.com/numpy/numpy)
    * Library used for array manipulation and data processing for Keras
* [Pandas](https://github.com/pandas-dev/pandas)
    * Data analysis tool used for data ingest and manipulation
* [TensorFlow](https://github.com/tensorflow/tensorflow)
    * The machine learning framework behind Keras used for sentiment analysis of tweets
* [tqdm](https://github.com/tqdm/tqdm)
    * Progress bar used for visualizing load times and model processing
* [Tweepy](http://docs.tweepy.org/en/latest/)
    * An open-source python library used to access the Twitter API
### Other
* [Twitter API](https://developer.twitter.com/en/docs)
    * Used to stream tweets live into the sentiment analysis model