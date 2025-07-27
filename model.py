import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HEADERS = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

def generate_answer(transcript, question):
    prompt = f"Based on the following transcript, answer the question:\n\nTranscript:\n{transcript}\n\nQuestion: {question}\n\nAnswer:"
    payload = {"inputs": prompt, "options": {"use_cache": False}}
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        return f"Error: {response.status_code} - {response.text}"
