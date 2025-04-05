import nltk
from razdel import tokenize
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
nltk.download('punkt_tab')
nltk.download('stopwords')

class Segmenter:
    del_ = ['.', '?', '!']

    def __init__(self, to_segment: str,):
        self.to_segment = to_segment

    def segment_noob(self):
        new = self.to_segment.split(self.del_[0])
        # for dl in self.del_:
        #     new = new.split(dl)
        return new

    def segment_nltk(self):
        self.to_segment = nltk.word_tokenize(self.to_segment)
        return self.to_segment

    def segment_natasha(self):
        return list(tokenize(self.to_segment))

    def segment(self):
        sentences = sent_tokenize(self.to_segment)

        # Загружаем Sentence-BERT
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Получаем эмбеддинги предложений
        sentence_embeddings = model.encode(sentences)

        # Вычисляем косинусное сходство между предложениями
        similarity_matrix = cosine_similarity(sentence_embeddings)

        # Определяем точки разрыва (там, где похожесть минимальна)
        threshold = 0.65  # Можно подобрать эмпирически ??????
        breakpoints = [i for i in range(len(sentences) - 1) if similarity_matrix[i, i + 1] < threshold]

        # Формируем смысловые части
        segments = []
        start = 0
        for bp in breakpoints:
            segments.append(" ".join(sentences[start: bp + 1]))
            start = bp + 1
        segments.append(" ".join(sentences[start:]))
        self.to_segment = segments
        return self.to_segment




