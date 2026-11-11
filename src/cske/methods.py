import re
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer
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
    """
    if df.empty or col_name not in df.columns:
        return []

    temp_series = df[col_name].dropna()
    if temp_series.empty:
        return []

    first_val = temp_series.iloc[0]
    if isinstance(first_val, str):
        try:
            sublists = [ast.literal_eval(x) for x in temp_series if isinstance(x, str)]
        except (ValueError, SyntaxError):
            sublists = []
    else:
        sublists = temp_series.tolist()

    flattened = [
        (item[0], float(item[1]))
        for sublist in sublists if isinstance(sublist, (list, tuple))
        for item in sublist if len(item) >= 2
    ]

    if not remove_duplicates:
        result = flattened
    else:
        unique_kws = {}
        for kw, score in flattened:
            if kw not in unique_kws or score > unique_kws[kw]:
                unique_kws[kw] = score
        result = list(unique_kws.items())

    if sort == 'descending':
        result.sort(key=lambda x: x[1], reverse=True)
    elif sort == 'ascending':
        result.sort(key=lambda x: x[1])

    return result

def count_occurrence(df: pd.DataFrame, words: List[Tuple], col_name: str) -> List[Tuple]:
    """Efficiently counts keyword occurrences in the dataframe."""
    if not words:
        return []

    corpus = df[col_name].dropna().astype(str).tolist()
    vocabulary = {str(item[0]).lower(): idx for idx, item in enumerate(words)}

    cv = CountVectorizer(vocabulary=vocabulary, lowercase=True)
    dtm = cv.fit_transform(corpus)
    
    counts = np.asarray(dtm.sum(axis=0)).flatten()
    return [(w[0], w[1], int(counts[vocabulary[str(w[0]).lower()]])) for w in words]

def keywords_avg_scores_kb(keyword_list: List[str], keyBERT, seed_keywords: List[str]):
    """Calculates the semantic distance between candidates and the seed centroid."""
    if not keyword_list:
        return []
    
    cand_emb = keyBERT.get_embeddings(keyword_list, show_progress=False)
    seed_emb = keyBERT.get_embeddings(seed_keywords, show_progress=False)
    
    centroid = np.mean(seed_emb, axis=0, keepdims=True)
    centroid_sim = cos_sim(cand_emb, centroid).cpu().numpy().flatten()
    
    matrix_sim = cos_sim(cand_emb, seed_emb).cpu().numpy()
    max_sim = np.max(matrix_sim, axis=1)

    avg_scores = (centroid_sim + max_sim) / 2    
    results = [(kw, round(float(s), 4)) for kw, s in zip(keyword_list, avg_scores)]
    return sorted(results, key=lambda x: x[1], reverse=True)