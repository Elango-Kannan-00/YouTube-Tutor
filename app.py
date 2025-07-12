import streamlit as st
from utils.transcript import get_transcript
from utils.embeddings import embed_and_store
from utils.qa import answer_question

st.set_page_config(page_title="AI-Powered YouTube Q&A")

st.title("🎓 AI-Powered YouTube Q&A Tutor")
st.markdown("Paste a YouTube video link to extract the transcript and ask questions based on the content.")

video_url = st.text_input("📺 Enter YouTube Video URL")

if st.button("🔍 Process Video"):
    with st.spinner("⏳ Extracting transcript..."):
        try:
            transcript = get_transcript(video_url)
            st.success("✅ Transcript Extracted!")
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.stop()

    with st.spinner("📦 Indexing for Q/A..."):
        index, sentences, model = embed_and_store(transcript)
        st.session_state.index = index
        st.session_state.sentences = sentences
        st.session_state.model = model
        st.success("✅ Ready to answer your questions!")

# Q&A Section
if "index" in st.session_state:
    question = st.text_input("❓ Ask a question about the video content:")
    if st.button("💬 Get Answer"):
        answer = answer_question(
            question,
            st.session_state.index,
            st.session_state.sentences,
            st.session_state.model
        )
        st.success(f"🤖 Answer: {answer}")
