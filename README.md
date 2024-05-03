# keyword_extractor
Code for class-specific keyword-extraction, for the submission to KONVENS 2024, titled: *An Improved Method for Class-specific Keyword Extraction: A Case Study in the German Business Registry*

## Example Usage

`import keyword_extractor

model = MODEL_CHECKPOINT # e.g., "deutsche-telekom/gbert-large-paraphrase-cosine"

EX = keyword_extractor.KeyExGen(embedding_model=model)

keywords = EX.keyword_pipeline(starting_seed=SEED_KEYWORDS, df=DATAFRAME, df_col_to_extract="column", topk=K, number_newseed=5, do_filter=False, n_iterations=5)`