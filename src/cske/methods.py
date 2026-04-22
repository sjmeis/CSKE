import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from collections import Counter
from .utils import cos_sim

def count_occurrence(df: pd.DataFrame, words: List[Tuple], col_name: str) -> List[Tuple]:
    """Efficiently counts keyword occurrences in the dataframe."""
    text_blob = " ".join(df[col_name].astype(str).str.lower())
    tokens = re.findall(r'\w+', text_blob)
    counts = Counter(tokens)
    return [(w[0], w[1], counts.get(w[0].lower(), 0)) for w in words]

def keywords_avg_scores_kb(keyword_list: List[str], keyBERT, seed_keywords: List[str]):
    """Calculates the semantic distance between candidates and the seed centroid."""
    cand_emb = keyBERT.model.encode(keyword_list)
    seed_emb = keyBERT.model.encode(seed_keywords)
    
    centroid = np.mean(seed_emb, axis=0, keepdims=True)
    
    centroid_sim = cos_sim(cand_emb, centroid).flatten()
    
    matrix_sim = cos_sim(cand_emb, seed_emb)
    max_sim = np.max(matrix_sim.numpy(), axis=1)
    
    avg_scores = (centroid_sim.numpy() + max_sim) / 2
    
    results = [(kw, round(float(s), 4)) for kw, s in zip(keyword_list, avg_scores)]
    return sorted(results, key=lambda x: x[1], reverse=True)