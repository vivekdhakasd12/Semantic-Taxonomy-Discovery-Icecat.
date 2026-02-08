from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score
import numpy as np
import pandas as pd
from . import config

def purity_score(y_true, y_pred):
    contingency_matrix = pd.crosstab(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix.values, axis=0)) / np.sum(contingency_matrix.values)

def compute_metrics(embeddings, labels, true_labels=None, sample_size=10000):
    """
    Computes a comprehensive dictionary of clustering metrics.
    Sampling used for silhouette score as it's O(N^2).
    """
    metrics = {}
    
    mask = labels != -1
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    metrics['n_clusters'] = n_clusters
    metrics['noise_ratio'] = np.sum(labels == -1) / len(labels)
    
    if n_clusters < 2:
        return metrics

    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        emb_sample = embeddings[idx]
        lab_sample = labels[idx]
    else:
        emb_sample = embeddings
        lab_sample = labels

    try:
        metrics['silhouette'] = silhouette_score(emb_sample, lab_sample)
        metrics['davies_bouldin'] = davies_bouldin_score(emb_sample, lab_sample)
        metrics['calinski_harabasz'] = calinski_harabasz_score(emb_sample, lab_sample)
    except Exception as e:
        print(f"Error computing unsupervised metrics: {e}")

    if true_labels is not None:
        metrics['purity'] = purity_score(true_labels, labels)
        metrics['ari'] = adjusted_rand_score(true_labels, labels)
        metrics['nmi'] = normalized_mutual_info_score(true_labels, labels)
        metrics['homogeneity'] = homogeneity_score(true_labels, labels)
        metrics['completeness'] = completeness_score(true_labels, labels)
        metrics['v_measure'] = v_measure_score(true_labels, labels)
        metrics['fowlkes_mallows'] = fowlkes_mallows_score(true_labels, labels)

    return metrics

