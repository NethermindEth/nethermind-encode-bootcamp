import os
import streamlit as st
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from langfuse.decorators import observe

# Set the browser title
st.set_page_config(page_title="Encode chat")

st.title("Echo Bot")

load_dotenv(".env")
load_dotenv(".env.langfuse")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add this function to get available models
def get_available_models():
    try:
        models = client.models.list()
        return [model.id for model in models.data if model.id.startswith("gpt")]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return ["gpt-3.5-turbo", "gpt-4"]  # Fallback to default models

# Add sidebar for model selection
st.sidebar.title("Model Selection")
available_models = get_available_models()
selected_model = st.sidebar.selectbox("Choose a model", available_models)

# Add sidebar for temperature selection
st.sidebar.title("Temperature")
temperature = st.sidebar.slider("Adjust temperature", min_value=0.0, max_value=2.0, value=0.5, step=0.1)

@observe()
def generate_answer(user_prompt):
    stream = stream_content_openai(user_prompt=user_prompt)
    return stream

@observe()
def handle_input(user_prompt):
    stream = generate_answer(user_prompt=user_prompt)
    response = st.write_stream(stream)
    return response

@observe(as_type="generation")
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
            model=selected_model,  # Use the selected model here
            temperature=temperature,  # Use the selected temperature here
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
