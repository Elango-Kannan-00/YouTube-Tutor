import streamlit as st
from transcript_extraction import get_transcript_chunks, get_embeddings
from gemini_qa import build_vector_store, get_top_chunks, ask_gemini
from dotenv import load_dotenv

load_dotenv()  # Loads environment variables

st.set_page_config(page_title="🎓 YouTube Q&A Assistant")

st.title("🎓 YouTube Video Q&A Assistant")
st.write("Ask questions based on the content of any YouTube video!")

youtube_url = st.text_input("📺 YouTube Video URL")
user_question = st.text_input("❓ Your Question")

if st.button("Get Answer"):
    if youtube_url and user_question:
        with st.spinner("Extracting transcript and generating answer..."):
            chunks = get_transcript_chunks(youtube_url)
            embeddings = get_embeddings(chunks)
            index = build_vector_store(embeddings)
            top_chunks = get_top_chunks(user_question, chunks, embeddings, index)
            answer = ask_gemini(user_question, top_chunks)
        st.success("✅ Answer:")
        st.markdown(f"> {answer}")
    else:
        st.warning("Please provide both a YouTube URL and a question.")
