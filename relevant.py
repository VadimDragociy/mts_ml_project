from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from segmenter import Segmenter
# from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel
from paraphrase import *

class Relevance:
    def __init__(self, text):
        self.text = text

        sgm = Segmenter(self.text)
        self.text = sgm.segment()
        # print(self.text)

        self.vectorizer = TfidfVectorizer()
        doc_vectors = self.vectorizer.fit_transform(self.text)
        self.knn = NearestNeighbors(n_neighbors=2, metric='cosine')
        self.knn.fit(doc_vectors)

    def search_for_query(self, query):
        query_vector = self.vectorizer.transform([query])
        distances, indices = self.knn.kneighbors(query_vector)

        # print("Наиболее релевантные фрагменты:")
        # for idx in indices[0]:
            # print(f"- {self.text[idx]}")


class Relevance_Frida:
    def __init__(self, search_document):
        sgm = Segmenter(search_document)
        self.search_document = sgm.segment()
        # print(self.search_document)

    def pool(self, hidden_state, mask, pooling_method="cls"):
        if pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif pooling_method == "cls":
            return hidden_state[:, 0]

    def search_for_query(self, search_query):
        tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRIDA")
        model = T5EncoderModel.from_pretrained("ai-forever/FRIDA")
        max_score = 0
        document = ""
        documents = []

        for partial_input in self.search_document:
            inputs = [
                f'search_query:{search_query}',
                #
                f'search_document:{partial_input}',
            ]

            tokenized_inputs = tokenizer(inputs, max_length=512, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                outputs = model(**tokenized_inputs)

            embeddings = self.pool(
                outputs.last_hidden_state,
                tokenized_inputs["attention_mask"],
                pooling_method="cls"  # or try "mean"
            )

            embeddings = F.normalize(embeddings, p=2, dim=1)
            sim_scores = embeddings[:1] @ embeddings[1:].T
            document = partial_input

            documents.append([sim_scores[0], document])
        documents.sort(reverse=True)
        answers = []
        for relevant_chunk in documents[:3]:
            # print(relevant_chunk[1])
            answer = paraphrase(relevant_chunk[1], search_query)
            answers.append(answer)
        return answers
