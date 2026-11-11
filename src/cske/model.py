import numpy as np
import pandas as pd
import torch
from typing import List, Union, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from .utils import cos_sim

class KeyBERTMod:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None, verbose: bool = True):
        self.verbose = verbose
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def get_embeddings(self, texts: Union[str, List[str]], show_progress: Optional[bool] = None) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()))

        display = show_progress if show_progress is not None else self.verbose
        return self.model.encode(
            texts,
            show_progress_bar=display,
            batch_size=64,
            convert_to_numpy=True
        )

    def extract_keywords_batch(
        self,
        docs: List[str],
        seed_keywords: List[str],
        seed_embeds: Optional[np.ndarray] = None,
        doc_weight: float = 0.0,
        seed_weight: float = 1.0,
        top_n: int = 5,
        vectorizer: Optional[CountVectorizer] = None,
        strategy: str = "hybrid"
    ) -> List[List[Tuple[str, float]]]:
        if not docs:
            return []

        if vectorizer is None:
            active_vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words="english")
            try:
                doc_term_mat = active_vectorizer.fit_transform(docs)
                feature_names = active_vectorizer.get_feature_names_out()
            except ValueError:
                return [[] for _ in docs]
        else:
            active_vectorizer = vectorizer
            try:
                if not hasattr(active_vectorizer, "vocabulary_"):
                    doc_term_mat = active_vectorizer.fit_transform(docs)
                else:
                    doc_term_mat = active_vectorizer.transform(docs)
                feature_names = active_vectorizer.get_feature_names_out()
            except ValueError:
                return [[] for _ in docs]

        if len(feature_names) == 0:
            return [[] for _ in docs]

        doc_embeds = self.get_embeddings(docs, show_progress=False)
        show_vocab_pbar = self.verbose and len(feature_names) > 1000
        vocab_embeds = self.get_embeddings(
            feature_names.tolist(), 
            show_progress=show_vocab_pbar
        )

        if seed_embeds is None:
            seed_embeds = self.get_embeddings(seed_keywords, show_progress=False)

        seed_sim_matrix = cos_sim(seed_embeds, vocab_embeds).cpu().numpy()

        if strategy == "max":
            seed_score = np.max(seed_sim_matrix, axis=0)
        elif strategy == "mean":
            seed_score = np.mean(seed_sim_matrix, axis=0)
        elif strategy == "hybrid":
            max_score = np.max(seed_sim_matrix, axis=0)
            mean_score = np.mean(seed_sim_matrix, axis=0)
            seed_score = (max_score + mean_score) / 2.0
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose 'max', 'mean', or 'hybrid'.")

        doc_sim_matrix = cos_sim(doc_embeds, vocab_embeds).cpu().numpy()

        total_weight = doc_weight + seed_weight
        if total_weight == 0:
            combined_scores = np.repeat(seed_score[np.newaxis, :], len(docs), axis=0)
        else:
            combined_scores = (doc_sim_matrix * doc_weight + seed_score * seed_weight) / total_weight

        results = []
        for row_idx in range(len(docs)):
            active_indices = doc_term_mat[row_idx].indices
            if len(active_indices) == 0:
                results.append([])
                continue

            row_scores = combined_scores[row_idx, active_indices]
            row_words = feature_names[active_indices]

            if len(row_scores) > top_n:
                top_k_idx = np.argpartition(row_scores, -top_n)[-top_n:]
                top_k_idx = top_k_idx[np.argsort(-row_scores[top_k_idx])]
            else:
                top_k_idx = np.argsort(-row_scores)

            results.append([(row_words[k], round(float(row_scores[k]), 4)) for k in top_k_idx])

        return results