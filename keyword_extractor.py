import pandas as pd
from key_tool_kit import KeyToolkit


class KeyExGen:
    
    def __init__(self, embedding_model: str):
        self.toolkit = KeyToolkit(embedding_model=embedding_model)

    def keyword_pipeline(
        self, 
        starting_seed, 
        df: pd.DataFrame,
        number_newseed, 
        vectorizer=None,
        df_col_to_extract='data',
        n_iterations=5, 
        doc_weight=0, 
        seed_weight=1,
        percentile_newseed_extraction = 99,
        count_occurrences = False,
        topk=None
        ):
        
        all_extracted_scores = self.toolkit.extract_keywords_iteration(
            starting_seed=starting_seed,
            df_whole=df,
            vectorizer=vectorizer,
            df_col_name=df_col_to_extract,
            split_n=n_iterations,
            doc_weight=doc_weight,
            seed_weight=seed_weight,
            percentile_newseed=percentile_newseed_extraction,
            number_newseed=number_newseed,
            count_occurrences=count_occurrences)
            
        extracted_filtered = self.toolkit.filter_extracted(all_extracted_scores, topk=topk)

        allwords = set(extracted_filtered + starting_seed)

        return allwords