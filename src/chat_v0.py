import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv


st.title("Echo Bot")

load_dotenv(".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(user_prompt):
    stream = stream_content_openai(user_prompt=user_prompt)
    return stream

def handle_input(user_prompt):
    stream = generate_answer(user_prompt=user_prompt)
    response = st.write_stream(stream)
    return response

def stream_content_openai(user_prompt):
    system_prompt = {"role": "user", "content": "You are a helpful assistant that sounds like Yoda from Star Wars."}
    try:
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]
        messages.append(system_prompt)
        messages.append({"role": "user", "content": user_prompt})
        print(messages)
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.5,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            yield chunk.choices[0].delta.content or ""
    except Exception as e:
        print(e)
        return None

# Suprising:
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

input = st.chat_input("Say something")
if input:
    with st.chat_message("user"):
        st.markdown(input)  
    with st.chat_message("assistant", avatar="👀"):
        response = handle_input(user_prompt=input)      
    st.session_state.messages.append({"role": "user", "content": input})
    st.session_state.messages.append({"role": "assistant", "content": response})
