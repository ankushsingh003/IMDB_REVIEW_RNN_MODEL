import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten,SimpleRNN
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import load_model



word_index = imdb.get_word_index()
index_word = {v : k for k, v in word_index.items()}

model = load_model('imdb_simpleRNN_model.h5')

def decode(text):
    return ' '.join([index_word.get(i - 3, '?') for i in text])

def preprocess_review(review):
    
    tokens = review.lower().split()
    
    indices = [word_index.get(word, 2) + 3 for word in tokens]  # 2 is for unknown words
    
    padded_sequence = pad_sequences([indices], maxlen=500)
    return padded_sequence

def predict_review(review):
    # Preprocess the review
    review_seq = []
    for word in review.split():
        index = word_index.get(word, 2)  # 2 is for unknown words
        review_seq.append(index + 3)  # Offset by 3 as per Keras IMDB dataset

    review_pad = pad_sequences([review_seq], maxlen=500)

    # Predict sentiment
    prediction = model.predict(review_pad)
    sentiment = "Positive" if prediction[0][0] >= 0.5 else "Negative"
    return sentiment, prediction[0][0]

# streamlit app
import streamlit as st
st.title("Movie Review Sentiment Analysis")
user_review = st.text_area("Enter your movie review here:")

if st.button("Predict Sentiment"):
    preprocess_input = preprocess_review(user_review)

    predict = model.predict(preprocess_input)
    sentiment = "Positive" if predict[0][0] >= 0.5 else "Negative"
    st.write(f"the prediction score is : {sentiment}")
    st.write(f"Confidence: {predict[0][0]:.4f}")

else: 
    st.write("Please enter a review and click the button to predict sentiment.")