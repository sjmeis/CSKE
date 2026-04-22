import numpy as np
import pandas as pd
import torch
from typing import List, Union, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from .utils import cos_sim

class KeyBERTMod:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

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