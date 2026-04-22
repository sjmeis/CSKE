<div align="left">

  [![PyPI version](https://img.shields.io/pypi/v/cske.svg)](https://pypi.org/project/cske/)
  [![License](https://img.shields.io/github/license/sjmeis/CSKE.svg)](https://github.com/sjmeis/CSKE/blob/main/LICENSE)

</div>

# CSKE: Class-Specific Keyword Extraction
CSKE is a high-performance Python library designed for iterative, class-specific keyword extraction. Unlike generic extractors, CSKE maintains coherence to a user-defined class, and leverages clustering techniques to ensure that expanded keyword sets remain semantically anchored to the target class.

This code was introduced in the KONVENS 2024 paper titled: *An Improved Method for Class-specific Keyword Extraction: A Case Study in the German Business Registry*

## Key Features
 - **Iterative Expansion**: Automatically discovers new keywords by walking through your dataset starting from a small seed list (defined by you!).
 - **Drift Prevention**: Uses clustering and filtering to weed out out "semantic drift" keywords that do not fit will to your defined domain.
 - **Weighted Extraction**: Balance the influence between the local document context and the global seed keywords.
 - **Hardware Accelerated**: Built on top of PyTorch and Sentence Transformers with automatic support for CUDA.

## Installation
```bash
pip install cske
```

## Usage
```python
import pandas as pd
from cske import CSKE

df = pd.DataFrame({
    "text_content": [
        "Neural networks are a subset of machine learning.",
        "The transformer architecture revolutionized NLP.",
        "Deep learning models require significant GPU resources.".
        "..."
    ]
})

# 2. Initialize the extractor, using any sentence transformer model, i.e., from Hugging Face
extractor = CSKE(embedding_model="all-MiniLM-L6-v2")

# 3. Run the pipeline
keywords = extractor.keyword_pipeline(
    starting_seed=["machine learning", "neural networks"],
    df=df,
    df_col_to_extract="text_content",
    n_iterations=3, # number of iterations (partitions of your data)
    number_newseed=2, # maximum number of "new" seeds per iteration
    do_filter=True  # whether to perform filtering to prevent drift
)

print(f"Expanded Keyword Set: {keywords}")
```

## Key Parameters, and what they mean

| Parameter     | Default     | Description |
| -----------   | ----------- | ----------- |
| `n_iterations`| `5`   | How many rounds of expansion to perform.
| `seed_weight` | `1.0` | Importance given to the original seed keywords.
| `doc_weight`  | `0.0` | Importance given to document context.
| `do_filter`   | `True`| Whether or not to apply filtering.
| `topk`        | `None` | If set, limits the final output to the top-k keywords.

---

## Citation
If you find `CSKE` useful or utilize it for your work, please considering citing:

```
@inproceedings{meisenbacher-etal-2024-improved,
    title = "An Improved Method for Class-specific Keyword Extraction: A Case Study in the {G}erman Business Registry",
    author = "Meisenbacher, Stephen  and
      Schopf, Tim  and
      Yan, Weixin  and
      Holl, Patrick  and
      Matthes, Florian",
    editor = "Luz de Araujo, Pedro Henrique  and
      Baumann, Andreas  and
      Gromann, Dagmar  and
      Krenn, Brigitte  and
      Roth, Benjamin  and
      Wiegand, Michael",
    booktitle = "Proceedings of the 20th Conference on Natural Language Processing (KONVENS 2024)",
    month = sep,
    year = "2024",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.konvens-main.18/",
    pages = "159--165"
}
```