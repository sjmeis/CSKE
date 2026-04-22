import numpy as np
import logging
import requests
from typing import List, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from sentence_transformers import SentenceTransformer
import umap

logger = logging.getLogger(__name__)

def min_dist(n, default):
    return max(2, min(n - 1, default))

class KeywordFilter:
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        self.model = SentenceTransformer(model_name)

    def _get_clusters(self, embeddings: np.ndarray, distance_threshold: float) -> np.ndarray:
        """Hierarchical clustering to group semantically similar keywords."""
        if len(embeddings) < 2:
            return np.zeros(len(embeddings))
        linkage_matrix = linkage(embeddings, method='ward')
        return fcluster(linkage_matrix, distance_threshold, criterion='distance')

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
        
        current_kws = [x[0] for x in keywords]
        seed_lower = [s.lower() for s in seed_keywords]

        logger.info(f"Filtering {len(current_kws)} keywords via Geometric Hull...")
        
        embeddings_all = self.model.encode(current_kws, batch_size=64)
        reducer = umap.UMAP(
            n_neighbors=min_dist(len(current_kws), 15), 
            min_dist=0.1, 
            n_components=5, 
            metric='cosine', 
            random_state=42
        )
        logger.info(f"Projecting {len(current_kws)} embeddings with UMAP...")
        embeddings_2d = reducer.fit_transform(embeddings_all)

        seed_indices = [i for i, txt in enumerate(texts) if txt.lower() in seed_lower]
        
        if len(seed_indices) < 3:
            logger.warning("Not enough seed keywords found in current set to form a Convex Hull.")
            return keywords, []

        seed_points = embeddings_2d[seed_indices]
        hull = ConvexHull(seed_points)
        hull_path = Path(seed_points[hull.vertices])

        is_inside = hull_path.contains_points(embeddings_2d, radius=0.05)
        
        filtered_keywords = [keywords[i] for i, inside in enumerate(is_inside) if inside]
        outliers = [current_kws[i] for i, inside in enumerate(is_inside) if not inside]

        logger.info(f"Hull filtering complete: {len(filtered_keywords)} kept, {len(outliers)} removed.")
        return filtered_keywords, outliers