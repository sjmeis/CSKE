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
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
    
    def get_embeddings(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        display = show_progress if show_progress is not None else self.verbose
        return self.model.encode(
            texts, 
            show_progress_bar=display, 
            batch_size=64,
            convert_to_numpy=True
        )

    def extract_keywords(
        self,
        doc: str,
        seed_keywords: List[str],
        doc_weight: float = 0.0,
        seed_weight: float = 1.0,
        top_n: int = 5,
        vectorizer: Optional[CountVectorizer] = None,
        **kwargs
    ) -> List[Tuple[str, float]]:
        """
        Improved keyword extraction with weighted seed influence.
        """
        if not doc or pd.isna(doc):
            return []

        if vectorizer is None:
            vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words="english")
        
        try:
            count = vectorizer.fit([doc])
            candidates = count.get_feature_names_out()
        except ValueError:
            return []

        doc_embed = self.get_embeddings([doc])
        cand_embeds = self.get_embeddings(candidates.tolist())
        seed_embeds = self.get_embeddings(seed_keywords)
        
        avg_seed_embed = np.mean(seed_embeds, axis=0, keepdims=True)

        target_embed = (doc_embed * doc_weight + avg_seed_embed * seed_weight) / (doc_weight + seed_weight)

        distances = cos_sim(target_embed, cand_embeds)
        results = sorted(zip(candidates, distances), key=lambda x: x[1], reverse=True)

        return [(word, round(float(score), 4)) for word, score in results[:top_n]]
    
    def extract_keywords_max(
        self,
        doc: str,
        seed_keywords: List[str],
        doc_weight: float = 0.0,
        seed_weight: float = 1.0,
        top_n: int = 5,
        vectorizer: Optional[CountVectorizer] = None
    ) -> List[Tuple[str, float]]:
        """
        Calculates the similarity of each candidate to every seed keyword 
        and picks the maximum similarity per candidate.
        """
        if not doc or pd.isna(doc):
            return []

        if vectorizer is None:
            vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words="english")
        
        try:
            count = vectorizer.fit([doc])
            candidates = count.get_feature_names_out()
        except ValueError:
            return []

        doc_embed = self.get_embeddings([doc], show_progress=False)
        cand_embeds = self.get_embeddings(candidates.tolist(), show_progress=False)
        seed_embeds = self.get_embeddings(seed_keywords, show_progress=False)

        from .utils import cos_sim
        similarity_matrix = cos_sim(seed_embeds, cand_embeds).numpy()
        
        max_seed_sim = np.max(similarity_matrix, axis=0)

        doc_sim = cos_sim(doc_embed, cand_embeds).numpy().flatten()

        combined_scores = (doc_sim * doc_weight + max_seed_sim * seed_weight) / (doc_weight + seed_weight)

        results = sorted(zip(candidates, combined_scores), key=lambda x: x[1], reverse=True)
        return [(word, round(float(score), 4)) for word, score in results[:top_n]]