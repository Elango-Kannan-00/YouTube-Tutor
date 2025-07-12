from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the FLAN-T5 base model for generative QA
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def answer_question(question, index, sentences, embedding_model):
    # Get top 5 context sentences from transcript using FAISS
    question_embedding = embedding_model.encode([question])
    _, I = index.search(question_embedding, k=5)
    context = " ".join([sentences[i] for i in I[0]])

    # Prompt for generating full answer
    prompt = f"""Answer the question clearly and simply based on the transcript below.

Transcript:
{context}

Question: {question}
Answer:"""

    # Tokenize and generate answer
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(**inputs, max_length=200)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer
