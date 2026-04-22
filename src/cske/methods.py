import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from collections import Counter
from .utils import cos_sim

def keywords_list_from_df_with_score(df: pd.DataFrame, col_name: str = 'keywords', sort: str = 'descending', remove_duplicates: bool = True):
    """
    Extracts and flattens keyword-score tuples from a DataFrame column.
    Handles columns stored as strings (via ast.literal_eval) or as native lists.
    """
    if df.empty or col_name not in df.columns:
        return []

    temp_df = df.dropna(subset=[col_name]).copy()

    try:
        first_val = temp_df[col_name].iloc[0]
        if isinstance(first_val, str):
            keywords_list = temp_df[col_name].apply(ast.literal_eval).tolist()
        else:
            keywords_list = temp_df[col_name].tolist()
    except (ValueError, SyntaxError, IndexError):
        keywords_list = temp_df[col_name].tolist()

    if not keywords_list:
        return []
    
    flattened_list = [
        (item[0], item[1]) 
        for sublist in keywords_list 
        if isinstance(sublist, list) 
        for item in sublist 
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]
    
    if remove_duplicates:
        unique_kws = {}
        for kw, score in flattened_list:
            if kw not in unique_kws or score > unique_kws[kw]:
                unique_kws[kw] = score
        flattened_data = list(unique_kws.items())
    else:
        flattened_data = flattened_list

    # Sorting logic
    if sort == 'ascending':
        return sorted(flattened_data, key=lambda x: x[1])
    elif sort == 'descending':
        return sorted(flattened_data, key=lambda x: x[1], reverse=True)
    
    return flattened_data

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