from transformers import pipeline
import streamlit as st

def summarize_text(text):
    # Load lightweight summarization model
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

    # Break long text into 1024-token chunks
    chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
    summary = ""

    # Progress bar to show status
    progress_bar = st.progress(0)
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        try:
            result = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
            summary += result[0]['summary_text'] + " "
            progress_bar.progress((i + 1) / total)
        except Exception as e:
            st.warning(f"⚠️ Skipping one chunk due to: {e}")
            continue

    progress_bar.empty()
    return summary.strip()
