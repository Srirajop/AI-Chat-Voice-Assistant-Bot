import os
import re
import uuid
import streamlit as st
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from TTS.api import TTS

# =========================
# ENV + SETUP
# =========================
load_dotenv()
os.makedirs("output", exist_ok=True)

st.set_page_config(page_title="Agentic AI Voice Assistant", page_icon="🤖")

# =========================
# TEXT CLEANING FOR TTS
# =========================
def clean_text_for_tts(text: str) -> str:
    text = re.sub(r"[*#=_>`\-]+", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text.strip()

def shorten_text(text: str, max_chars=300):
    if len(text) > max_chars:
        return text[:max_chars] + "."
    return text

# =========================
# LOAD TTS (ONCE)
# =========================
@st.cache_resource
def load_tts():
    return TTS(model_name="tts_models/en/ljspeech/fast_pitch")

tts = load_tts()

def speak(text: str) -> str:
    safe_text = shorten_text(clean_text_for_tts(text))
    filename = f"speech_{uuid.uuid4().hex}.wav"
    path = os.path.join("output", filename)
    tts.tts_to_file(text=safe_text, file_path=path)
    return path

# =========================
# LOAD AGENT (ONCE)
# =========================
@st.cache_resource
def load_agent():
    return Agent(
        model=Groq(id="llama-3.1-8b-instant"),
        markdown=False,
        instructions="""
        You are a friendly AI assistant.
        Remember previous messages and respond conversationally.
        Keep answers clear and concise.
        """
    )

agent = load_agent()

# =========================
# SESSION MEMORY
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# UI
# =========================
st.title("🤖 AI Chat + Voice Assistant")
st.caption("Memory-enabled • Persistent Voice • Built with Agno")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "audio" in msg:
            st.audio(msg["audio"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Build conversation context
    conversation = ""
    for msg in st.session_state.messages:
        conversation += f"{msg['role']}: {msg['content']}\n"

    # Agent response
    response = agent.run(conversation).content

    # Generate voice
    audio_path = speak(response)

    # Store assistant message WITH audio
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "audio": audio_path
        }
    )

    with st.chat_message("assistant"):
        st.markdown(response)
        st.audio(audio_path)

