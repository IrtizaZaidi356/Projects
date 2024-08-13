import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disables oneDNN optimizations
import tensorflow 

# Set the page configuration
st.set_page_config(
    page_title="Text Emotion APP",  # Title of the app
    page_icon=":smiley:",  # You can use emoji, a local file path, or a URL to an image
    layout="wide",  # Optional: Use "wide" for a wider layout, or "centered" for a centered layout
    initial_sidebar_state="expanded"  # Optional: Use "expanded" or "collapsed" for the sidebar state
)

# Load the saved model and tokenizer
model = load_model("./model/best_model.h5")

with open("./model/tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

with open("./model/label_encoder.pkl", "rb") as label_encoder_file:
    label_encoder = pickle.load(label_encoder_file)

with open("./model/model_params.pkl", "rb") as model_params_file:
    model_params = pickle.load(model_params_file)

max_length = model_params['max_length']

# Streamlit App
st.title("Text Emotion Classification")

# User input
input_text = st.text_area("Enter text to classify its emotion:")

if st.button("Predict Emotion"):
    if input_text:
        # Preprocess the input text
        input_sequence = tokenizer.texts_to_sequences([input_text])
        padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
        
        # Predict the emotion
        prediction = model.predict(padded_input_sequence)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
        
        # Display the predicted emotion
        st.write(f"The predicted emotion is: **{predicted_label[0]}**")
    else:
        st.write("Please enter some text to classify.")


st.write("\nText Emotion App Version 1.0\n")

st.header("Developed by: SYED IRTIZA ABBAS ZAIDI")