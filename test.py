from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

documents = [
    "Python — это мощный язык программирования.",
    "Frida используется для динамического анализа приложений.",
    "Машинное обучение помогает анализировать текст."
]

doc_embeddings = model.encode(documents)
query = "анализ текста"
query_embedding = model.encode(query)

scores = util.cos_sim(query_embedding, doc_embeddings)[0]
best_match = documents[scores.argmax()]

print(f"Наиболее релевантный фрагмент: {best_match}")
