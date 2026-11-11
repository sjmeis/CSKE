import numpy as np
import logging
import requests
from typing import List, Tuple, Optional
from scipy.spatial import ConvexHull, QhullError
from matplotlib.path import Path
from sentence_transformers import SentenceTransformer
import umap

logger = logging.getLogger(__name__)

def min_dist(n, default):
    """Ensures n_neighbors is always at least 2 and strictly less than n."""
    return max(2, min(n - 1, default))

class KeywordFilter:
    def __init__(self, model_name: str = 'all-mpnet-base-v2', shared_model: Optional[SentenceTransformer] = None, device: Optional[str] = None):
        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = SentenceTransformer(model_name, device=device)

    def get_synonyms(self, keywords: List[str]) -> List[str]:
        """Fetches related terms from ConceptNet to expand the filter's 'safety zone'."""
        synonyms = set()
        for kw in keywords:
            word = "_".join(kw.split(" ")).lower()
            try:
                res = requests.get(f"http://api.conceptnet.io/query?node=/c/en/{word}&rel=/r/RelatedTo", timeout=5)
                if res.status_code == 200:
                    edges = res.json().get('edges', [])
                    for edge in edges:
                        synonyms.add(edge['start']['label'])
            except Exception as e:
                logger.warning(f"Could not fetch synonyms for {kw}: {e}")
        return list(synonyms)

    def filter(self, keywords: List[Tuple[str, float]], seed_keywords: List[str]) -> Tuple[List[Tuple], List[str]]:
        if not keywords: return [], []

        if len(keywords) < 5:
            logger.warning("Too few keywords to perform geometric filtering. Returning all.")
            return keywords, []
        
        current_kws = [x[0] for x in keywords]
        seed_clean = list(dict.fromkeys(s.strip() for s in seed_keywords if s.strip()))
        seed_clean_lower = {s.lower() for s in seed_clean}

        current_kws_lower = {k.lower() for k in current_kws}
        missing_seeds = [s for s in seed_clean if s.lower() not in current_kws_lower]
        
        all_texts = current_kws + missing_seeds
        n_samples = len(all_texts)

        logger.info(f"Filtering {len(current_kws)} keywords via Geometric Hull...")
        
        embeddings_all = self.model.encode(all_texts, batch_size=64, show_progress_bar=False)
        reducer = umap.UMAP(
            n_neighbors=min_dist(n_samples, default=15),
            min_dist=0.1, 
            n_components=2, 
            metric='cosine', 
            random_state=42,
            verbose=False
        )
        logger.info(f"Projecting {n_samples} embeddings with UMAP...")
        embeddings_2d = reducer.fit_transform(embeddings_all)
        cand_embeddings_2d = embeddings_2d[:len(current_kws)]

        seed_indices = [
            i for i, txt in enumerate(all_texts) 
            if txt.lower() in seed_clean_lower
        ]
        
        if len(seed_indices) < 3:
            logger.warning("Not enough seed keywords found in current set to form a Convex Hull.")
            return keywords, []

        seed_points = embeddings_2d[seed_indices]

        unique_seed_points = np.unique(seed_points, axis=0)
        if len(unique_seed_points) < 3:
            logger.warning("Fewer than 3 unique 2D coordinates for seeds. Skipping hull.")
            return keywords, []

        try:
            hull = ConvexHull(unique_seed_points)
            hull_vertices_2d = unique_seed_points[hull.vertices]
            hull_path = Path(hull_vertices_2d)

            # Check candidate containment inside polygon with a slight safety boundary
            is_inside = hull_path.contains_points(cand_embeddings_2d, radius=0.05)

            filtered_keywords = [keywords[i] for i, inside in enumerate(is_inside) if inside]
            outliers = [current_kws[i] for i, inside in enumerate(is_inside) if not inside]

            logger.info(f"Hull filtering complete: {len(filtered_keywords)} kept, {len(outliers)} removed.")
            return filtered_keywords, outliers

        except QhullError as e:
            logger.warning(f"ConvexHull generation failed (seeds may be collinear in 2D space): {e}. Returning all.")
            return keywords, []