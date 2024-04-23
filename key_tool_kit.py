import pandas as pd
import numpy as np
from typing import List, Tuple
from key_bert_mod import KeyBERTMod
from utils import sort_keywords_list, damping_func, keywords_only
from methods import keywords_list_from_df_with_score, count_occurrence, keywords_avg_scores_kb
import swifter


class KeyToolkit:
    
    def __init__(self, embedding_model):
        self.extractor = KeyBERTMod(model=embedding_model)
        
    def extract_keywords_iteration(
            self,
            starting_seed,
            df_whole: pd.DataFrame,
            vectorizer,
            df_col_name,
            split_n = 5, 
            doc_weight =0, 
            seed_weight=1,
            percentile_newseed = 99, 
            number_newseed = 3, 
            count_occurrences = False
        ) -> List[Tuple[str, float, int]]:
        
        df_whole.dropna(subset=[df_col_name], inplace=True)

        dfs = np.array_split(df_whole, split_n)
        
        seed = [item.lower() for item in starting_seed]

        kw_dict_all = {kw:1 for kw in seed}
        
        for df, i in zip(dfs, range(split_n)):
            print("Iteration {}".format(i))
            # foreach row in the dataset we do
            df['max_keyword'] = df.swifter.progress_bar(enable=True).apply(lambda x: 
                self.extractor.extract_keywords_max(x[df_col_name],
                vectorizer = vectorizer, 
                seed_keywords=seed, 
                doc_weight=doc_weight, 
                seed_weight=seed_weight), axis=1)
        
            max_kw_list = dict(keywords_list_from_df_with_score(df, 'max_keyword'))

            df['mean_keyword'] = df.swifter.progress_bar(enable=True).apply(lambda x: 
                self.extractor.extract_keywords(x[df_col_name],
                vectorizer = vectorizer, 
                seed_keywords=seed, 
                doc_weight=doc_weight, 
                seed_weight=seed_weight), axis=1)

            mean_kw_list = dict(keywords_list_from_df_with_score(df, 'mean_keyword'))

            combined_kw = list(max_kw_list.keys()) + list(mean_kw_list.keys())

            combined_kw_scores = keywords_avg_scores_kb(combined_kw, keyBERT=self.extractor, seed_keywords=seed)

            if isinstance(combined_kw_scores, dict):
                kw_extracted = list(combined_kw_scores.items())
            elif isinstance(combined_kw_scores, list):
                kw_extracted = combined_kw_scores

            kw_extracted.sort(key=lambda x: x[1], reverse=True)
            kw_dict_all.update(kw_extracted)

            numbers = [t[1] for t in kw_extracted]
            percentile = np.percentile(numbers, percentile_newseed)
            top_results = []
            fallback_results = []

            for kw, score in kw_extracted:
                if score > percentile and kw not in seed:
                    top_results.append(kw)
                elif len(fallback_results) < number_newseed and kw not in seed:
                    fallback_results.append(kw)

            if not top_results:
                top_results = fallback_results

            seed.extend(top_results)
            seed = list(set(seed))

        df_combined = pd.concat(dfs)

        for item in starting_seed:
            kw_dict_all[item.lower()] = 1.0

        keywords = list(kw_dict_all.items())

        if count_occurrences:
            keywords = count_occurrence(df_combined, keywords)
        
        keywords = sort_keywords_list(keywords)

        return keywords

    def filter_extracted(
        self, 
        keyword_list, 
        damping_k = 5, 
        damping_alpha = 0.0001,
        start_position = 0,
        topk=None):
        
        keyword_list = sort_keywords_list(keyword_list)
        if topk is not None:
            kwonly = keywords_only(keyword_list[:topk])
        else:
            number = round(damping_func(len(keyword_list), damping_k, damping_alpha))
            kwonly = keywords_only(keyword_list[start_position:number+start_position])

        return kwonly