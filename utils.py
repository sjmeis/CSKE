import numpy as np
import torch
from torch import Tensor
from typing import List, Tuple
from sklearn.feature_extraction.text import CountVectorizer


def get_unigrams(list: list):
    return [word for word in list if len(word.split()) == 1]

def damping_func(n, k, alpha):
    return k * (np.log((n)) - np.log(alpha))    

def sort_keywords_list(kwlist, sort_by_index:int = 1):
    return sorted(kwlist, key=lambda x: x[sort_by_index], reverse=True)

def keywords_only(keywords: List[Tuple], loc = 0):
    if not keywords:
        return []
    return [str(t[loc]) for t in keywords]

def keep_top_percentile(wordlist: List[Tuple], scores_index:int, top_percentile: float):

    if not isinstance(wordlist, list):
        raise ValueError("wordlist must be a list.")
    
    if not all(isinstance(item, tuple) for item in wordlist):
        raise ValueError("All elements in the list must be tuples.")
    
    tuple_lengths = {len(t) for t in wordlist}
    if len(tuple_lengths) > 1:
        raise ValueError("All tuples in the list must have the same length.")
    
    if not (0 <= top_percentile <= 100):
        raise ValueError("the top_percentile has to be between 0 and 100")

    scores = [item[scores_index] for item in wordlist]
    percentile = np.percentile(scores, top_percentile)
    top_wordlist = [(word, score) for word, score in wordlist if score > percentile]
    return top_wordlist

def cos_sim(a: Tensor, b: Tensor):
    
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

