import os
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx 
from openai import OpenAI
from langfuse import Langfuse
from langfuse.decorators import langfuse_context, observe
from langfuse.openai import openai

st.set_page_config(page_title="Awesome Chat")
st.title("Awesome Chat")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
langfuse = Langfuse()

ctx = get_script_run_ctx()

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar settings
st.sidebar.title("Settings")
model = st.sidebar.selectbox("Choose OpenAI model", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"], index=2)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1)
traits = st.sidebar.text_input("Enter traits", value="funny, grumpy")
if rating := st.sidebar.radio(
    "Rate the last answer",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: ":star:" * x,
    horizontal=True
):
    if 'trace_id' in st.session_state:
        print(f'trace_id={st.session_state.trace_id} rating={rating}')
        langfuse.score(
            name="feedback-on-trace",
            value=rating,
            trace_id=st.session_state.trace_id
        )

# Print message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

@observe()
def handle_input(user_prompt):
    stream = generate_answer(user_prompt=user_prompt)
    response = st.write_stream(stream)
    return response

@observe()
def generate_answer(user_prompt):
    langfuse_context.update_current_trace(
        session_id=ctx.session_id,
        tags=[model, "encode-bootcamp"],
        user_id="test_user"
    )
    stream = stream_content_openai(user_prompt=user_prompt)
    return stream

@observe(as_type="generation")
def stream_content_openai(user_prompt):
    st.session_state.trace_id = langfuse_context.get_current_trace_id()
    prompt = langfuse.get_prompt("personality-prompt", type="chat")
    system_prompt = prompt.compile(traits=traits)[0]

    langfuse_context.update_current_observation(        
        prompt=prompt,
    )
    try:
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]
        messages.append(system_prompt)
        messages.append({"role": "user", "content": user_prompt})
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            stream=True,
        )
        for chunk in response:
            yield chunk.choices[0].delta.content or ""
    except Exception as e:
        langfuse_context.update_current_trace(
            metadata={"error": str(e)}
        )
        print(e)
        return None


# Handle prompt input
if input := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(input)    

    with st.chat_message("assistant", avatar="👀"):
        response = handle_input(user_prompt=input)        

    st.session_state.messages.append({"role": "user", "content": input})
    st.session_state.messages.append({"role": "assistant", "content": response})