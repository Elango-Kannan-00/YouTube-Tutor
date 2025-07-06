from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_question(question, index, sentences, embedding_model):
    question_embedding = embedding_model.encode([question])
    D, I = index.search(question_embedding, k=5)
    context = " ".join([sentences[i] for i in I[0]])
    result = qa_pipeline(question=question, context=context)
    return result['answer']