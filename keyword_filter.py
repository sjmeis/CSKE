import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer, util
import requests

class KeywordFilter:

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            
    # Method to generate clusters
    def keywordstoClusters(self, keyword_embeddings, distance_threshold, eps=0.5, density_threshold=0.1):
        linkage_matrix = linkage(keyword_embeddings, method='ward')
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        unique_clusters = np.unique(cluster_labels)
        combined_labels = np.zeros_like(cluster_labels)
        for cluster_id in unique_clusters:
            cluster_points = keyword_embeddings[cluster_labels == cluster_id]
            # Check density
            distance_matrix = pairwise_distances(cluster_points, metric='euclidean')
            if((len(cluster_points) * (len(cluster_points) - 1) / 2)==0):
                density = 0
            else:
                density = len(distance_matrix[distance_matrix < eps]) / (len(cluster_points) * (len(cluster_points) - 1) / 2)
            if density < density_threshold and density!=0:
                # Recursive call for further clustering
                sub_cluster_labels = self.keywordstoClusters(cluster_points, distance_threshold/1.1, eps, density_threshold)
                combined_labels[cluster_labels == cluster_id] = sub_cluster_labels + max(combined_labels) + 1
            else:
                combined_labels[cluster_labels == cluster_id] = cluster_labels[cluster_labels == cluster_id]
        return combined_labels

    # Get synonyms for the seed keywords
    def get_synonyms_for_keywords(self, keywords):
        synonyms = set()

        for keyword in keywords:
            word = "_".join(keyword.split(" ")).lower()
            url = f"http://api.conceptnet.io/query?node=/c/en/{word}&rel=/r/RelatedTo"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for edge in data['edges']:
                    start = edge['start']['label']
                    synonyms.add(start)
            else:
                return "Error: Unable to fetch data"
        return list(synonyms)

    def filter(self, keywords, seed_keywords):
        print("Filtering ({})...".format(len(keywords)))
        orig_keywords = keywords
        keywords = [x[0] for x in keywords]

        print("\tClustering...")
        embeddings = np.array(self.model.encode(keywords))
        cluster = self.keywordstoClusters(embeddings, len(embeddings))
            
        unique_cluster_values = np.unique(cluster)
        filter_keywords = [5] #just for initialization
        main_keywords = keywords
        radius = 0
        
        print("\tConvex Hull...")
        while (filter_keywords != []): 
            unique_cluster_values = np.unique(cluster)
            filtered_keywords = []
            clusters_contain_seeds = []
            embeddingsTemp = np.array(self.model.encode(keywords))
            if(len(embeddingsTemp) <= 50):
                tsne = TSNE(n_components=2, random_state=42, perplexity=len(embeddingsTemp)-1)
            else:
                tsne = TSNE(n_components=2, random_state=42)
            embeddings = tsne.fit_transform(embeddingsTemp)
            for target_cluster in unique_cluster_values:
                cluster_keywords = [keywords[i].lower() for i in range(len(keywords)) if cluster[i] == target_cluster]
                for seed in seed_keywords:
                    if(seed.lower() in cluster_keywords):
                        clusters_contain_seeds.append(target_cluster)
                        break
    
            # Find convex hull of the selected clusters
            selected_cluster_indices = np.isin(cluster, clusters_contain_seeds)
            selected_cluster_points = embeddings[selected_cluster_indices]
            hull = ConvexHull(selected_cluster_points)

            # Identify clusters inside the convex hull
            hull_path = Path(selected_cluster_points[hull.vertices] )
            radius = radius + 0.0001
            embed_hull_bool = hull_path.contains_points(embeddings, radius = radius)     
            filtered_keywords = [value for value, is_true in zip(keywords, embed_hull_bool) if is_true]
            cluster = [cluster[i] for i, keyword in enumerate(keywords) if keyword in filtered_keywords]
            filter_keywords = list(set(keywords)-set(filtered_keywords))
            keywords = filtered_keywords

            if(filter_keywords!=[]):
                embeddingsTemp = np.array([self.model.encode(keyword) for keyword in keywords])
                cluster = self.keywordstoClusters(embeddingsTemp, len(embeddingsTemp))
            
        out_keywords = list(set(main_keywords)-set(filtered_keywords))
        clusters_outliers_perc_per_class = len(out_keywords)/len(main_keywords)

        print("\tLOF + ISO...")
        # Outlier detection using Local Outlier Factor and Isolation Forest
        if(clusters_outliers_perc_per_class != 0.0):
            embeddings = np.array(self.model.encode(keywords))
            # Fit the Local Outlier Factor model
            lof = LocalOutlierFactor(contamination=0.5)  # Adjust contamination based on the expected percentage of outliers
            outlier_scores_lof = lof.fit_predict(embeddings)
            # Fit the Isolation Forest model
            iso = IsolationForest(contamination=0.5)  # Adjust contamination based on the expected percentage of outliers
            outlier_scores_iso = iso.fit_predict(embeddings)

            # Identify outliers based on LOF scores
            outliers_lof = np.where(outlier_scores_lof == -1)[0]
            keywords_outliers_lof = [keywords[i] for i in outliers_lof]

            # Identify outliers based on ISO scores
            outliers_iso = np.where(outlier_scores_iso == -1)[0]
            keywords_outliers_iso = [keywords[i] for i in outliers_iso]

            merge_all_outliers_with_duplicates = keywords_outliers_lof + keywords_outliers_iso

        # Union of clustering and outlier detection results
        outlier_union = set(out_keywords + merge_all_outliers_with_duplicates)

        print("\tCos Sim w/ Synonyms...")
        # cosine similarity for further outliers filtering
        seeds_plus = seed_keywords + self.get_synonyms_for_keywords(seed_keywords) 
        seed_embeddings = self.model.encode(seeds_plus)
        final_outliers = []
        for word in outlier_union:
            word_embedding = self.model.encode(word)
            score = util.cos_sim(word_embedding, seed_embeddings).tolist()[0]
            score = [(x + 1) / 2 for x in score]
            top_id = np.argsort(score)[::-1][0]
            if(score[top_id]<=0.8):
                final_outliers.append(word.lower())

        keywords_wo_outliers = [x for x in orig_keywords if x[0] not in final_outliers]
        print("Finished ({} outliers).".format(len(final_outliers)))
        return keywords_wo_outliers, final_outliers
