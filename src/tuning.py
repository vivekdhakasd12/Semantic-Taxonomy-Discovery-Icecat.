import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from . import clustering, config
import time

def tune_hyperparameters(embeddings, algo_name, param_grid, sample_size=10000):
    """
    Grid search for the best hyperparameters.
    
    Args:
        embeddings: feature vectors
        algo_name: string name of the algorithm
        param_grid: dict of parameter lists, e.g. {'eps': [0.3, 0.5], 'min_samples': [5]}
        sample_size: size of sample for Silhouette Score calculation (speed optimization)
        
    Returns:
        best_params: dict
        best_score: float
        detailed_results: list of dicts
    """
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n--- Tuning {algo_name} ({len(combinations)} combinations) ---")
    
    best_score = -2.0 # Silhouette ranges from -1 to 1
    best_params = None
    results = []
    
    if len(embeddings) > sample_size:
        idx = np.random.choice(len(embeddings), sample_size, replace=False)
        val_embeddings = embeddings[idx]
    else:
        val_embeddings = embeddings

    for i, params in enumerate(combinations):
        print(f"   [{i+1}/{len(combinations)}] Testing params: {params}...", end="\r")
        
        try:
            start_t = time.time()
            labels = None
            
            
            if algo_name == 'DBSCAN':
                labels = clustering.run_dbscan(embeddings, **params, quiet=True)
            elif algo_name == 'KMeans':
                 labels = clustering.run_kmeans(embeddings, **params, quiet=True)
            elif algo_name == 'MiniBatchKMeans':
                 labels = clustering.run_minibatch_kmeans(embeddings, **params, quiet=True)
            elif algo_name == 'OPTICS':
                labels = clustering.run_optics(embeddings, **params, quiet=True)
            elif algo_name == 'BIRCH':
                labels = clustering.run_birch(embeddings, **params, quiet=True)
            elif algo_name == 'BisectingKMeans':
                labels = clustering.run_bisecting_kmeans(embeddings, **params, quiet=True)
            elif algo_name == 'HDBSCAN':
                 labels = clustering.run_hdbscan(embeddings, **params, quiet=True)
            
            if labels is not None:
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters < 2 or n_clusters > len(embeddings) - 1:
                    score = -1.0 # Invalid
                else:
                    if len(embeddings) > sample_size:
                        val_labels = labels[idx]
                    else:
                        val_labels = labels
                        
                    score = silhouette_score(val_embeddings, val_labels)
                
                elapsed = time.time() - start_t
                
                res = {
                    'params': params,
                    'n_clusters': n_clusters,
                    'silhouette': score,
                    'time': elapsed
                }
                results.append(res)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
        except Exception as e:
            print(f"\n      Failed with {params}: {e}")

    print(f"\n   > Best Score: {best_score:.4f} with {best_params}")
    return best_params, best_score, results
