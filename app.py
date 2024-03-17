# Import necessary libraries
import streamlit as st
import numpy as np 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from normalizer import normalize

# Set the page configuration
st.set_page_config(
    page_title="Banglish to Bangla Translation App",  # Title of the app displayed in the browser tab
    page_icon=":shield:",  # Path to a favicon or emoji to be displayed in the browser tab
    initial_sidebar_state="auto"  # Initial state of the sidebar ("auto", "expanded", or "collapsed")
)


# Load custom CSS styling
with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
     
# Function to load the pre-trained model
@st.cache_resource(experimental_allow_widgets=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglat5", use_fast=True) # Set legacy=False
    model = AutoModelForSeq2SeqLM.from_pretrained("Soyeda10/BanglishToBangla") # Set legacy=False
    return tokenizer, model


# Load the tokenizer and model
tokenizer, model = get_model()



# Add a header to the Streamlit app
st.header("BengaliBridge")

# Add placeholder text with custom CSS styling
st.markdown("<span style='color:black'>Enter your Banglish text here</span>", unsafe_allow_html=True)

# Text area for user input with label and height set to 250
user_input = st.text_area("Enter your Banglish text here", "", height=250, label_visibility="collapsed")

# Button for submitting the input
submit_button = st.button("Translate")

# Perform prediction when user input is provided and the submit button is clicked
if user_input and submit_button:
    input_ids = tokenizer(normalize(user_input), padding=True, truncation=True, max_length=128, return_tensors="pt").input_ids
    generated_tokens = model.generate(input_ids, max_new_tokens=128)  # Set max_new_tokens to control generation length
    decoded_tokens = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    st.write(f"<span style='color:black'>Bangla Translation: {decoded_tokens}</span>", unsafe_allow_html=True)



