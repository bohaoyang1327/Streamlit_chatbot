import streamlit as st
from transformers import pipeline

st.title("GPT-2 Chatbot")

model = pipeline("text-generation", model="gpt2-medium", framework="pt")

user_input = st.text_input("Your message:")
if user_input:
    response = model(
    user_input, 
    max_length=150, 
    temperature=0.8, 
    top_k=50, 
    top_p=0.95, 
    num_return_sequences=1
)
    st.write(response[0]['generated_text'])
