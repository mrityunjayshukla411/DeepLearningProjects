import numpy as np
import pandas as pd
import streamlit as st
# import seaborn as sns 
import string
# import matplotlib.pyplot as plt
# from matplotlib import colors
import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_text as text
# from wordcloud import WordCloud

# # Preprocessing and evaluation
import nltk
from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelBinarizer
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
nltk.download("stopwords")
nltk.download("omw-1.4")
nltk.download("wordnet")
import os

path = os.path.dirname(__file__)
import pickle

with open(path + 'tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open(path + 'lemmatizer.pickle', 'rb') as handle:
    lb = pickle.load(handle)

def cleaning(text):
    #remove punctuations and uppercase
    clean_text = text.translate(str.maketrans('','',string.punctuation)).lower()
    
    #remove stopwords
    clean_text = [word for word in clean_text.split() if word not in stopwords.words('english')]
    
    #lemmatize the word
    sentence = []
    for word in clean_text:
        lemmatizer = WordNetLemmatizer()
        sentence.append(lemmatizer.lemmatize(word, 'v'))

    return ' '.join(sentence)

model_lstm = tf.keras.models.load_model(path + '/Model_LSTM/')

def lstm_prediction(text):
    clean_text = cleaning(text)
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq)

    pred = model_lstm.predict(padded)
    # Get the label name back
    result = lb.inverse_transform(pred)[0]
    
    return result

st.title('Spotify Review Sentiment Analysis')

text = st.text_input('Write a review of the spotify app in less than 500 words')
if text:
    x = lstm_prediction(text)
    st.markdown(f'## Your Feelings toward Spotify are:- `{x}`')
    # st.write('Your Feelings toward Spotify are:- ', x)



