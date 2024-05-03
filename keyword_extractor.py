import pandas as pd
from key_tool_kit import KeyToolkit
from keyword_filter import KeywordFilter

class KeyExGen:
    
    def __init__(self, embedding_model: str):
        self.toolkit = KeyToolkit(embedding_model=embedding_model)
        self.filterer = KeywordFilter()

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
        topk=None,
        do_filter=True,
        keep_scores=False
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
        
        if do_filter == True:
            filtered, outliers = self.filterer.filter(keywords=all_extracted_scores, seed_keywords=starting_seed)
        else:
            filtered = all_extracted_scores
        
        extracted_filtered = self.toolkit.filter_extracted(filtered, topk=topk, keep_scores=keep_scores)

        starting_seed = [x.lower() for x in starting_seed]
        if keep_scores == True:
            allwords = [x for x in extracted_filtered if x[0] not in starting_seed]
        else:
            allwords = [x for x in extracted_filtered if x not in starting_seed]

        return allwords