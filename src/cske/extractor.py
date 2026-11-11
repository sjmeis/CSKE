import pandas as pd
import logging
from typing import List, Optional, Union

from .toolkit import KeyToolkit
from .filter import KeywordFilter

logger = logging.getLogger("cske.extractor")

class CSKE:
    """
    Class-specific Keyword Extraction (CSKE).
    
    This class provides the main pipeline for iteratively extracting and filtering 
    keywords based on starting seeds and a target dataset.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", verbose: bool = True, filter_model: Optional[str] = None):
        """
        Initialize the CSKE pipeline.
        """
        self.verbose = verbose
        pkg_logger = logging.getLogger("cske")
        pkg_logger.setLevel(logging.INFO if self.verbose else logging.ERROR)

        self.toolkit = KeyToolkit(embedding_model=embedding_model, verbose=self.verbose)
        if filter_model:
            self.filterer = KeywordFilter(model_name=filter_model)
        else:
            self.filterer = KeywordFilter(shared_model=self.toolkit.extractor.model)

    def keyword_pipeline(
        self, 
        starting_seed: List[str], 
        df: pd.DataFrame,
        number_newseed: int, 
        vectorizer=None,
        df_col_to_extract: str = 'data',
        n_iterations: int = 5, 
        doc_weight: float = 0.0, 
        seed_weight: float = 1.0,
        percentile_newseed_extraction: float = 99.0,
        count_occurrences: bool = False,
        topk: Optional[int] = None,
        do_filter: bool = True,
        keep_scores: bool = False,
        strategy: str = "hybrid"
    ) -> Union[List[str], List[tuple]]:
        """
        Executes the full keyword extraction, expansion, and filtering pipeline.
        
        Args:
            starting_seed: Initial list of keywords to guide the extraction.
            df: The input DataFrame containing text data.
            number_newseed: Number of new keywords to add to the seed each iteration.
            df_col_to_extract: The column name in the DataFrame to process.
            topk: If set, returns exactly the top K keywords.
            do_filter: Whether to apply the Convex Hull / Outlier filtering.
            keep_scores: If True, returns (keyword, score) tuples.
            
        Returns:
            A list of extracted keywords or tuples.
        """
        if not starting_seed:
            raise ValueError("starting_seed cannot be empty.")
        if df.empty or df_col_to_extract not in df.columns:
            raise ValueError(f"DataFrame must contain column '{df_col_to_extract}'.")
        
        logger.info(f"Starting CSKE pipeline with {len(starting_seed)} seed words.")

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
            count_occurrences=count_occurrences,
            strategy=strategy
        )
        
        if do_filter and all_extracted_scores:
            logger.info("Applying Convex Hull and Outlier Filtering...")
            filtered, _ = self.filterer.filter(
                keywords=all_extracted_scores, 
                seed_keywords=starting_seed
            )
        else:
            filtered = all_extracted_scores
        
        extracted_filtered = self.toolkit.filter_extracted(
            filtered, 
            topk=topk, 
            keep_scores=keep_scores
        )

        starting_seed_lower = [x.lower() for x in starting_seed]
        
        if keep_scores:
            allwords = [x for x in extracted_filtered if x[0].lower() not in starting_seed_lower]
        else:
            allwords = [x for x in extracted_filtered if x.lower() not in starting_seed_lower]

        logger.info(f"Pipeline complete. Extracted {len(allwords)} new keywords.")
        return allwords