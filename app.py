import streamlit as st
import os
from dotenv import load_dotenv

from utils.downloader import download_audio
from utils.transcriber import transcribe_audio
from utils.summarizer import summarize_text
from utils.vector_store import store_documents, create_qa_chain


# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="YouTube Video Summarizer",
    layout="wide"
)

st.title("🎥 YouTube Video Summarizer + Q&A (Whisper + Groq + DeepLake)")

st.markdown("---")

# Input URL
url = st.text_input("Enter YouTube Video URL")

# Generate Summary Button
if st.button("Generate Summary"):

    if not url:
        st.warning("Please enter a YouTube URL.")
    else:

        # Step 1: Download Audio
        with st.spinner("📥 Downloading audio..."):
            audio_path = download_audio(url)

        # Step 2: Transcribe with Whisper
        with st.spinner("🎙 Transcribing with Whisper..."):
            transcript = transcribe_audio(audio_path)

        # Step 3: Summarize
        with st.spinner("🧠 Generating Summary using Groq..."):
            summary, docs = summarize_text(transcript)

        st.subheader("📌 Video Summary")
        st.write(summary)

        # Step 4: Store in DeepLake
        with st.spinner("📦 Creating Vector Database..."):
            db = store_documents(docs)
            qa_chain = create_qa_chain(db)

        # Save QA chain in session
        st.session_state["qa_chain"] = qa_chain

        st.success("✅ Ready for Q&A!")

st.markdown("---")

# Q&A Section
if "qa_chain" in st.session_state:

    st.subheader("💬 Ask Questions About the Video")

    question = st.text_input("Enter your question")

    if st.button("Ask"):

        if question:
            with st.spinner("🔍 Searching and Generating Answer..."):
                answer = st.session_state["qa_chain"].invoke({"question": question})

            st.write("### 📝 Answer")
            st.write(answer.content if hasattr(answer, 'content') else answer)
        else:
            st.warning("Please enter a question.")