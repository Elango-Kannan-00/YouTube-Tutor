from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_transcript(video_url):
    video_id = re.search(r"v=([a-zA-Z0-9_-]+)", video_url).group(1)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([entry['text'] for entry in transcript])

def split_text(text, max_tokens=200):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def get_transcript_chunks(video_url):
    text = extract_transcript(video_url)
    return split_text(text)

def get_embeddings(chunks):
    return model.encode(chunks)
