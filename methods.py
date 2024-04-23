import ast
import re
import numpy as np
import pandas as pd

from utils import cos_sim
from typing import Counter, List, Tuple, Union
from key_bert_mod import KeyBERTMod


def keywords_list_from_df_with_score(df, col_name='keywords', sort='descending', remove_duplicates = True):

    df.dropna(subset=[col_name], inplace=True)

    try:
        keywords_list = df[col_name].apply(ast.literal_eval).tolist()

    except (ValueError, SyntaxError): 
        keywords_list = df[col_name].tolist()

    if keywords_list is None:
        return []
    
    flattened_list = [(item[0], item[1]) for sublist in keywords_list if sublist is not None for item in sublist]
    
    if remove_duplicates:
        flattened_set = set(flattened_list)
    else:
        flattened_set = flattened_list

    if sort == 'ascending':
        return sorted(flattened_set, key=lambda x: x[1])
    elif sort == 'descending':
        return sorted(flattened_set, key=lambda x: x[1], reverse=True)
    else:
        return list(flattened_set)

def count_occurrence(df: pd.DataFrame, words: Union[str, List[str], List[Tuple]], col_name: str = 'purpose'):

    df_list = df[col_name].astype(str).str.lower().tolist()

    if (isinstance(words, list) or isinstance(words, set)) and all(isinstance(i, tuple) for i in words):
        words_only = {word[0] for word in words}
        word_counter = Counter(word for text in df_list for word in re.sub(r'[^\w\s]', ' ', text).split() if word in words_only) #substituting anything that is NOT a letter or a number with a space
        tuples_with_count = [(item[0], item[1], word_counter[item[0]]) for item in words]

        return tuples_with_count
    elif isinstance(words, str):
        words = [words]
    
    word_counter = {word: 0 for word in words}

    word_counter.update(Counter(word for text in df_list for word in re.sub(r'[^\w\s]', ' ', text).split() if word in words)) #substituting anything that is NOT a letter or a number with a space
    
    return list(word_counter)

def keywords_avg_scores_kb(keyword_set, keyBERT: KeyBERTMod = None, seed_keywords = None, seed_embeddings = None):
    keyword_list = list(keyword_set)
    if keyBERT == None:
        keyBERT = KeyBERTMod("deutsche-telekom/gbert-large-paraphrase-cosine")

    keyword_emb = keyBERT.model.encode(keyword_list)

    if seed_embeddings is None:
        if seed_keywords is not None:
                if isinstance(seed_keywords[0], str):
                    seed_embeddings = keyBERT.model.encode(seed_keywords)
                else:
                    raise ValueError("Seed keywords should be strings")
        else:
            raise ValueError("No seed keywords or embeddings provided!") 

    mean_emb = seed_embeddings.mean(axis=0, keepdims=True)       
    mean_similarity = cos_sim(keyword_emb, mean_emb)

    similarity_matrix = cos_sim(seed_embeddings, keyword_emb)
    max_rows_per_col = np.argmax(similarity_matrix, axis=0)
    cols = np.arange(len(max_rows_per_col))
    max_similarity = [similarity_matrix[i,j] for i, j in zip(max_rows_per_col, cols)]
    
    averaged_scores = [(mean_similarity[i] + max_similarity[i]) / 2 for i in range(len(keyword_list))]

    keywords = [(keyword, round(float(score), 4)) for keyword, score in zip(keyword_list, averaged_scores)]
    keywords.sort(key = lambda x: x[1], reverse = True)

    return keywords