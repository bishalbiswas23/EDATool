import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the generative AI model
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
chat_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

# Set up Streamlit app
st.title('EDA Tool')

# Define chat interface
# user_input = st.text_input('User:', value='', max_chars=100)
user_input = st.text_input('User:', value='', max_chars=100, key='user_input')

chat_history = []

# Function to generate AI response
def generate_response(input_text):
    chat_history.append(input_text)
    response = chat_generator(chat_history)
    latest_turn = response[-1]  # Get the most recent conversation turn
    generated_text = latest_turn[0]['generated_text']  # Access the generated text
    return generated_text

# Perform EDA tasks based on user input
def perform_eda(data):
    # Add EDA code here
    pass

# Main chat loop
while user_input:
    if st.button('Send', key='send_button'):
        # Add user query to chat history
        chat_history.append(user_input)

        # Generate AI response
        ai_response = generate_response(user_input)
        st.text_area('AI:', value=ai_response, height=100)

        # Perform EDA tasks if required
        if 'analyze' in user_input.lower():
            # Upload and preprocess data
            uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
            if uploaded_file:
                data = pd.read_csv(uploaded_file)
                # perform_eda(data)

                # Display analysis results
                st.subheader('Data Analysis Results')
                st.write(data.head())
                st.pyplot()  # Display Matplotlib or Seaborn plots here
            else:
                st.warning('Please upload a CSV file.')
        else:
            st.info('Please provide a query to perform analysis.')

    user_input = st.text_input('User:', value='', max_chars=100)

st.info('Enter a query to get started.')
