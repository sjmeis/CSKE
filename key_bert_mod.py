import numpy as np
from typing import List, Union, Tuple

from sklearn import __version__ as sklearn_version
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from packaging import version
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available

class KeyBERTMod:
    
    def __init__(self, model="sentence-transformers/all-MiniLM-L6-v2"):
        if is_available():
            self.model = SentenceTransformer(model, device="cuda")
        else:
            self.model = SentenceTransformer(model)

    def extract_keywords(
        self,
        doc: str,
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        vectorizer: CountVectorizer = None,
        seed_keywords: Union[List[str], List[List[str]]] = None,
        doc_embeddings: np.array = None,
        word_embeddings: np.array = None,
        seed_weight: int = 1,
        doc_weight: int = 0
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        
        if isinstance(doc, str):
            if doc:
                doc = [doc]
            else:
                return []

        if vectorizer:
            count = vectorizer.fit(doc)
        else:
            try:
                count = CountVectorizer(
                    ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    min_df=min_df,
                    vocabulary=candidates,
                ).fit(doc)
            except ValueError:
                return []

        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()
        df = count.transform(doc)

        if word_embeddings is not None:
            if word_embeddings.shape[0] != len(words):
                raise ValueError("Make sure that the `word_embeddings` are generated from the function "
                                 "`.extract_embeddings`. \nMoreover, the `candidates`, `keyphrase_ngram_range`,"
                                 "`stop_words`, and `min_df` parameters need to have the same values in both "
                                 "`.extract_embeddings` and `.extract_keywords`.")

        if doc_embeddings is None:
            doc_embeddings = self.model.encode(doc)
            
        if word_embeddings is None:
            word_embeddings = self.model.encode(words)

        if seed_keywords is not None:

            if isinstance(seed_keywords[0], str):
                seed_embeddings = self.model.encode(seed_keywords).mean(axis=0, keepdims=True)   
            elif len(doc) != len(seed_keywords):
                raise ValueError("The length of docs must match the length of seed_keywords")
            else:
                seed_embeddings = np.vstack([
                    self.model.encode(keywords).mean(axis=0, keepdims=True)
                    for keywords in seed_keywords
                ])
            doc_embeddings = ((doc_embeddings * doc_weight + seed_embeddings * seed_weight) / (doc_weight+seed_weight))

        all_keywords = []
        for index, _ in enumerate(doc):

            try:
                candidate_indices = df[index].nonzero()[1]
                candidates = [words[index] for index in candidate_indices]
                candidate_embeddings = word_embeddings[candidate_indices]
                doc_embedding = doc_embeddings[index].reshape(1, -1)

            
                distances = cosine_similarity(doc_embedding, candidate_embeddings)
                keywords = [
                    (candidates[index], round(float(distances[0][index]), 4)) 
                        for index in distances.argsort()[0][-top_n:]
                ][::-1]
                    

                all_keywords.append(keywords)

            except ValueError:
                all_keywords.append([])

        if len(all_keywords) == 1:
            all_keywords = all_keywords[0]

        return all_keywords
    
    def extract_keywords_max(
        self,
        doc: str,
        seed_keywords: List[str],
        candidates: List[str] = None,
        keyphrase_ngram_range: Tuple[int, int] = (1, 1),
        stop_words: Union[str, List[str]] = "english",
        top_n: int = 5,
        min_df: int = 1,
        vectorizer: CountVectorizer = None,
        doc_embeddings: np.array = None,
        word_embeddings: np.array = None,
        seed_weight: int = 1,
        doc_weight: int = 0
    ) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        
        if isinstance(doc, str):
            if doc:
                doc = [doc]
            else:
                return []

        if vectorizer:
            count = vectorizer.fit(doc)
        else:
            try:
                count = CountVectorizer(
                    ngram_range=keyphrase_ngram_range,
                    stop_words=stop_words,
                    min_df=min_df,
                    vocabulary=candidates,
                ).fit(doc)
            except ValueError:
                return []

        if version.parse(sklearn_version) >= version.parse("1.0.0"):
            words = count.get_feature_names_out()
        else:
            words = count.get_feature_names()
        df = count.transform(doc)

        if word_embeddings is not None:
            if word_embeddings.shape[0] != len(words):
                raise ValueError("Make sure that the `word_embeddings` are generated from the function "
                                 "`.extract_embeddings`. \nMoreover, the `candidates`, `keyphrase_ngram_range`,"
                                 "`stop_words`, and `min_df` parameters need to have the same values in both "
                                 "`.extract_embeddings` and `.extract_keywords`.")

        if doc_embeddings is None:
            doc_embeddings = self.model.encode(doc)
        if word_embeddings is None:
            word_embeddings = self.model.encode(words)


        if seed_keywords is not None:
            if isinstance(seed_keywords[0], str):
                seed_embeddings = self.model.encode(seed_keywords)

            elif len(doc) != len(seed_keywords):
                raise ValueError("The length of docs must match the length of seed_keywords")
            else:
                seed_embeddings = np.vstack([
                    self.model.encode(keywords).mean(axis=0, keepdims=True)
                    for keywords in seed_keywords
                ])

        all_keywords = []
        for index, _ in enumerate(doc):

            try:
                candidate_indices = df[index].nonzero()[1]
                candidates = [words[index] for index in candidate_indices]
                candidate_embeddings = word_embeddings[candidate_indices]
                doc_embedding = doc_embeddings[index].reshape(1, -1)                    
                    
                sim_to_doc = cosine_similarity(doc_embedding, candidate_embeddings)

                similarity_matrix = cosine_similarity(seed_embeddings, candidate_embeddings)
                max_rows_per_col = np.argmax(similarity_matrix, axis=0)

                cols = np.arange(len(max_rows_per_col))

                everything = [(seed_keywords[i], candidates[j], similarity_matrix[i,j]) for i, j in zip(max_rows_per_col, cols)]
                everything.sort(key=lambda x: x[2], reverse=True)
                

                top_n_keywords = everything[:top_n]

                keywords = []
                for seed_keyword, candidate_keyword, seed_similarity in top_n_keywords:
                    candidate_index = candidates.index(candidate_keyword)
                    doc_similarity = sim_to_doc[0][candidate_index]
                    weighted_avg_similarity = ((doc_weight * doc_similarity) + (seed_weight * seed_similarity)) / (doc_weight + seed_weight)
                    keywords.append((candidate_keyword, round(weighted_avg_similarity, 4)))

                all_keywords.append(keywords)

            except ValueError:
                all_keywords.append([])

        if len(all_keywords) == 1:
            all_keywords = all_keywords[0]

        return all_keywords