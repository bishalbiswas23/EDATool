import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the generative AI model
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
chat_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

uploaded_file = any
data = any

st.title("EDA Chat Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Function to generate AI response
def generate_response(input_text):
    response = chat_generator(input_text)
    # latest_turn = response[0]  # Get the most recent conversation turn
    generated_text = response[0]['generated_text']  # Access the generated text
    return generated_text

def ai_response():
    response = generate_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    check_history()

def check_history():
    st.info(st.session_state)


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Please enter your query..."):
    # Display user message in chat message container
    st.chat_message("user").write(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    check_history()

    analysis = ['analyze', 'analysis', 'analyse', 'data analysis']
    head = ['head','top5','top 5']

    if any(keyword in prompt.lower() for keyword in analysis):
        # Upload and preprocess data
        with st.chat_message("assistant"):
            uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
            
            check_history()
            if uploaded_file is not None:
                check_history()
                df = pd.read_csv(uploaded_file)
                st.dataframe(df)
                check_history()
                st.session_state.messages.append({"role": "assistant", "content": df})
                
            else:
                check_history()
                st.session_state.messages.append({"role": "assistant", "content": "couldn't upload"})
                
            #     st.info("abcd")
            #     print("abccd")
            #     data = pd.read_csv(uploaded_file)
               

            #     # Display analysis results
            #     st.subheader('Data Analysis Results')
            #     st.write(data.head())
            # st.session_state.messages.append({"role": "assistant", "content": uploaded_file})
            # st.info()
    
  
        
        
        # with st.chat_message("assistant"):
        #     st.write(type(uploaded_file))
        # st.session_state.messages.append({"role": "assistant", "content": uploaded_file})
        
        

    # elif any(keyword in prompt.lower() for keyword in head):
    #     response = type(data)
    #     with st.chat_message("assistant"):
    #         st.markdown(response)
    #     st.session_state.messages.append({"role": "assistant", "content": response})
    
            

    else:
        ai_response()


# import streamlit as st
# import pandas as pd

# def main():
#     st.title("CSV File Uploader")

#     # File uploader
#     uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

#     if uploaded_file is not None:
#         # Read the CSV file data
#         df = pd.read_csv(uploaded_file)

#         # Display the data as plain text
#         st.dataframe(df)

#     with st.chat_message("assistant"):
#         st.write("Hello 👋")
#         st.dataframe(df)

# if __name__ == "__main__":
#     main()