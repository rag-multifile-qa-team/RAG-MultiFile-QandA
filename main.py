# main.py (Checklist-complete version: Ollama + RAG + Multilingual + Voice In/Out)

import os
import tempfile

import streamlit as st
import torch

from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama

from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
import pyttsx3



# Page config (MUST be first Streamlit call)

st.set_page_config(page_title="Ask RAG - Multi-file Support", page_icon="*", layout="wide")
st.title("Ask RAG - Multi-file Support")



# Device detection

def get_device() -> str:
    # CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        return "cuda"

    # Apple Silicon (MPS)
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        return "mps"

    return "cpu"


def get_whisper_runtime(device: str):
   
    if device == "cuda":
        return "cuda", "float16"

    # mps or cpu -> cpu int8 (stable)
    return "cpu", "int8"


DEVICE = get_device()
TORCH_DEVICE = torch.device(DEVICE)



# Sidebar: Checklist + Settings

with st.sidebar:
    st.header("Checklist")
    st.markdown("- LLM (Ollama)")
    st.markdown("- GenAI (RAG)")
    st.markdown("- Hugging Face (Embeddings - all-MiniLM-L6-v2)")
    st.markdown("- Multilingual (TR/EN/FR/ES)")
    st.markdown("- Voice input  (Mic + Whisper)")
    st.markdown("- Voice output (Offline TTS)")

    # Device badge
    if DEVICE == "cuda":
        st.success(f" Device: {DEVICE.upper()}")
    elif DEVICE == "mps":
        st.info(f" Device: {DEVICE.upper()}")
    else:
        st.warning(f" Device: {DEVICE.upper()}")

    st.divider()
    st.subheader("Settings")

    # Language control
    language = st.selectbox("Response Language", ["Türkçe", "English", "Français", "Español"], index=0)
    force_lang_instruction = True  # always enforce

    # Ollama settings
    ollama_model = st.text_input("Ollama model", value="gemma3:4b")
    ollama_base_url = st.text_input("Ollama base_url", value="http://localhost:11434")

    # Voice settings
    st.caption("Voice Input/Output")
    enable_voice_input = st.toggle("Voice input (mic)", value=True)
    enable_voice_output = st.toggle("Voice output (TTS)", value=True)

    # Whisper settings
    whisper_size = st.selectbox("Whisper model size", ["small", "medium"], index=0)
    whisper_device, whisper_compute = get_whisper_runtime(DEVICE)
    st.caption(f"🎙️ Whisper runtime: {whisper_device.upper()} / {whisper_compute}")



# Embeddings (device-aware)  

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": DEVICE},  # cuda / cpu / mps
    encode_kwargs={"normalize_embeddings": True},
)



# Cached Whisper model (single source of truth) 

@st.cache_resource
def load_whisper_model(size: str, device: str, compute_type: str):
    return WhisperModel(size, device=device, compute_type=compute_type)


whisper_model = load_whisper_model(whisper_size, whisper_device, whisper_compute)



# Setup LLM (Ollama)

llm = Ollama(
    model=ollama_model.strip(),
    base_url=ollama_base_url.strip()
)



# Modification: Multilingual instruction

def add_language_instruction(user_text: str) -> str:
    if not force_lang_instruction:
        return user_text

    instructions = {
        "Türkçe": "Lütfen cevabı Türkçe ver. (Yanıt net, kısa ve anlaşılır olsun.)",
        "English": "Please answer in English. (Be clear, concise, and accurate.)",
        "Français": "Veuillez répondre en français. (Réponse claire, concise et précise.)",
        "Español": "Por favor responde en español. (Respuesta clara, concisa y precisa.)",
    }
    return f"{user_text}\n\n{instructions.get(language, instructions['English'])}"



# Modification: Voice output (TTS)

def tts_to_wav_bytes(text: str) -> bytes:
    engine = pyttsx3.init()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        wav_path = f.name

    try:
        engine.save_to_file(text, wav_path)
        engine.runAndWait()
        with open(wav_path, "rb") as rf:
            return rf.read()
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass



# Modification: Voice input (mic -> whisper)

def audio_bytes_to_wav_file(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        return f.name


def transcribe_audio_whisper(audio_bytes: bytes) -> str:
    wav_path = audio_bytes_to_wav_file(audio_bytes)
    try:
        segments, _info = whisper_model.transcribe(wav_path)
        return "".join(seg.text for seg in segments).strip()
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass



# Upload multiple files

uploaded_files = st.file_uploader(
    "Upload files (PDF, DOCX, TXT, CSV)",
    type=["pdf", "docx", "txt", "csv"],
    accept_multiple_files=True
)



# Load and process multiple files (RAG index)  uses embedding above

@st.cache_resource
def load_files(files):
    if not files:
        return None, []

    loaders = []
    temp_files = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name
            temp_files.append(temp_path)

        if file.name.endswith(".pdf"):
            loaders.append(PyPDFLoader(temp_path))
        elif file.name.endswith(".txt"):
            loaders.append(TextLoader(temp_path, encoding="utf-8"))
        elif file.name.endswith(".docx"):
            loaders.append(UnstructuredWordDocumentLoader(temp_path))
        elif file.name.endswith(".csv"):
            loaders.append(CSVLoader(temp_path))

    index = VectorstoreIndexCreator(
        embedding=embedding,  #  device-aware embedding
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50),
    ).from_loaders(loaders)

    return index, temp_files


if uploaded_files:
    index, temp_files = load_files(uploaded_files)
else:
    index, temp_files = None, []



# Chat state

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])



# Main Q/A chain

if index:
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
        input_key="question"
    )

    voice_text = ""
    if enable_voice_input:
        st.caption("🎙️ Voice Input")
        mic_result = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt=" Stop",
            key="mic"
        )

        if mic_result and mic_result.get("bytes"):
            with st.spinner("Transcribing audio..."):
                voice_text = transcribe_audio_whisper(mic_result["bytes"])
            if voice_text:
                st.success("Transcription ready ")
                st.write(voice_text)

    prompt = st.chat_input("Enter your prompt (or use voice input)")
    if (not prompt) and voice_text:
        prompt = voice_text

    if prompt:
        prompt_with_lang = add_language_instruction(prompt)

        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            response = chain.run(prompt_with_lang)

        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

        if enable_voice_output:
            with st.expander("🔊 Voice Output", expanded=False):
                if st.button("Read last answer aloud"):
                    with st.spinner("Generating speech..."):
                        audio_wav = tts_to_wav_bytes(response)
                    st.audio(audio_wav, format="audio/wav")

else:
    st.warning("Please upload files to start querying.")



# Clear All

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Clear All"):
        st.session_state.messages = []
        for file_path in temp_files:
            try:
                os.remove(file_path)
            except Exception:
                pass
        st.rerun()

with col2:
    st.caption("Tip: Upload files first → then ask questions (text or voice).")
