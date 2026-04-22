import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Union, Optional
from tqdm.auto import tqdm
tqdm.pandas()

from .model import KeyBERTMod
from .utils import sort_keywords_list, damping_func, keywords_only
from .methods import keywords_list_from_df_with_score, count_occurrence, keywords_avg_scores_kb

logger = logging.getLogger(__name__)

class KeyToolkit:
    """
    Core utilities for iterative keyword extraction and list management.
    """
    
    def __init__(self, embedding_model: str, verbose: bool = True):
        self.extractor = KeyBERTMod(model_name=embedding_model, verbose=self.verbose)
        
    def extract_keywords_iteration(
        self,
        starting_seed: List[str],
        df_whole: pd.DataFrame,
        vectorizer,
        df_col_name: str,
        split_n: int = 5, 
        doc_weight: float = 0.0, 
        seed_weight: float = 1.0,
        percentile_newseed: float = 99, 
        number_newseed: int = 3, 
        count_occurrences: bool = False
    ) -> List[Tuple[str, float, int]]:
        """
        Runs the iterative expansion logic by splitting the dataframe and 
        evolving the seed set.
        """

        df_work = df_whole.copy()
        df_work.dropna(subset=[df_col_name], inplace=True)

        dfs = np.array_split(df_work, split_n)
        current_seed = [item.lower() for item in starting_seed]

        kw_dict_all = {kw: 1.0 for kw in current_seed}
        
        for i, df_batch in enumerate(dfs):
            logger.info(f"Processing Iteration {i+1}/{split_n}...")
            
            apply_func = df_batch[df_col_name].progress_apply if self.verbose else df_batch[df_col_name].apply
            df_batch['max_keyword'] = df_batch.apply_func(
                lambda x: self.extractor.extract_keywords_max(
                    x[df_col_name],
                    vectorizer=vectorizer, 
                    seed_keywords=current_seed, 
                    doc_weight=doc_weight, 
                    seed_weight=seed_weight
                ), axis=1
            )
            
            max_kw_list = dict(keywords_list_from_df_with_score(df_batch, 'max_keyword'))
            
            combined_kw_scores = keywords_avg_scores_kb(
                list(max_kw_list.keys()), 
                keyBERT=self.extractor, 
                seed_keywords=current_seed
            )
            
            kw_dict_all.update(dict(combined_kw_scores))

            scores_only = [t[1] for t in combined_kw_scores]
            threshold = np.percentile(scores_only, percentile_newseed)
            
            new_seeds = [
                kw for kw, score in combined_kw_scores 
                if score >= threshold and kw not in current_seed
            ]

            if not new_seeds:
                new_seeds = [
                    kw for kw, _ in combined_kw_scores[:number_newseed] 
                    if kw not in current_seed
                ]

            current_seed.extend(new_seeds[:number_newseed])
            current_seed = list(set(current_seed)) # Deduplicate

        df_final = pd.concat(dfs)
        keywords = list(kw_dict_all.items())

        if count_occurrences:
            keywords = count_occurrence(df_final, keywords, col_name=df_col_name)
        
        return sort_keywords_list(keywords)

    def filter_extracted(
        self, 
        keyword_list: List[Tuple], 
        damping_k: int = 5, 
        damping_alpha: float = 0.0001,
        start_position: int = 0,
        topk: Optional[int] = None,
        keep_scores: bool = False
    ) -> Union[List[str], List[Tuple]]:
        """
        Final filtering using either a hard Top-K limit or a logarithmic 
        damping function based on total discovered keywords.
        """
        keyword_list = sort_keywords_list(keyword_list)
        
        if topk is not None:
            final_selection = keyword_list[:topk]
        else:
            # use logarithmic damping to decide how many keywords to keep
            num_to_keep = round(damping_func(len(keyword_list), damping_k, damping_alpha))
            final_selection = keyword_list[start_position : num_to_keep + start_position]

        if keep_scores:
            return final_selection
        return keywords_only(final_selection)