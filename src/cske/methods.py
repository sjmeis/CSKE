import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from collections import Counter
import ast
from .utils import cos_sim

def keywords_list_from_df_with_score(
    df: pd.DataFrame, 
    col_name: str = 'keywords', 
    sort: str = 'descending', 
    remove_duplicates: bool = True
) -> List[Tuple[str, float]]:    
    """
    Extracts and flattens keyword-score tuples from a DataFrame column.
    Handles columns stored as strings (via ast.literal_eval) or as native lists.
    """
    if df.empty or col_name not in df.columns:
        return []

    temp_series = df[col_name].dropna()
    if temp_series.empty:
        return []

    first_val = temp_series.iloc[0]
    if isinstance(first_val, str):
        try:
            keywords_list = temp_series.apply(ast.literal_eval).tolist()
        except (ValueError, SyntaxError):
            keywords_list = temp_series.tolist()
    else:
        keywords_list = temp_series.tolist()

    flattened_list = [
        (item[0], float(item[1]))
        for sublist in keywords_list
        if isinstance(sublist, (list, tuple))
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

    if sort == 'ascending':
        return sorted(flattened_data, key=lambda x: x[1])
    elif sort == 'descending':
        return sorted(flattened_data, key=lambda x: x[1], reverse=True)

    return flattened_data

def count_occurrence(df: pd.DataFrame, words: List[Tuple], col_name: str) -> List[Tuple]:
    """Efficiently counts keyword occurrences in the dataframe."""
    corpus = df[col_name].dropna().astype(str).str.lower().tolist()
    results = []
    
    for item in words:
        kw = str(item[0]).lower()
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b')
        count = sum(len(pattern.findall(text)) for text in corpus)
        results.append((item[0], item[1], count))
        
    return results

def keywords_avg_scores_kb(keyword_list: List[str], keyBERT, seed_keywords: List[str]):
    """Calculates the semantic distance between candidates and the seed centroid."""
    if not keyword_list:
        return []
    
    cand_emb = keyBERT.get_embeddings(keyword_list, show_progress=False)
    seed_emb = keyBERT.get_embeddings(seed_keywords, show_progress=False)
    
    centroid = np.mean(seed_emb, axis=0, keepdims=True)
    centroid_sim = cos_sim(cand_emb, centroid).cpu().numpy().flatten()
    
    matrix_sim = cos_sim(cand_emb, seed_emb).cpu().numpy()
    max_sim = np.max(matrix_sim.numpy(), axis=1)

    avg_scores = (centroid_sim.numpy() + max_sim) / 2    
    results = [(kw, round(float(s), 4)) for kw, s in zip(keyword_list, avg_scores)]
    return sorted(results, key=lambda x: x[1], reverse=True)