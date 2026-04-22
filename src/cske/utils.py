import numpy as np
import torch
from typing import List, Tuple, Union

def damping_func(n: int, k: int, alpha: float) -> float:
    """Logarithmic damping: $ f(n) = k \cdot (\ln(n) - \ln(\alpha)) $"""
    if n <= 0: return 0
    return k * (np.log(n) - np.log(alpha))

def cos_sim(a: Union[np.ndarray, torch.Tensor], b: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Standardized Cosine Similarity using PyTorch for speed."""
    if not isinstance(a, torch.Tensor): a = torch.tensor(a)
    if not isinstance(b, torch.Tensor): b = torch.tensor(b)
    
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def sort_keywords_list(kwlist: List[Tuple], index: int = 1):
    return sorted(kwlist, key=lambda x: x[index], reverse=True)

def keywords_only(keywords: List[Tuple]) -> List[str]:
    return [str(t[0]) for t in keywords]